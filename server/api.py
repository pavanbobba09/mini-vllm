"""OpenAI-compatible completions server.

POST /v1/completions with the official openai client works unmodified against
base_url=http://localhost:8000/v1, streaming (SSE) and non-streaming.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from engine.config import EngineConfig
from engine.scheduler import Request, Scheduler

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

_FINISHED = object()  # queue sentinel


@dataclass
class _Subscriber:
    loop: asyncio.AbstractEventLoop
    queue: "asyncio.Queue[Union[int, Tuple[object, str]]]"
    finish_reason: str = "length"
    token_ids: List[int] = field(default_factory=list)


class EngineWorker:
    """Owns the scheduler on a dedicated thread; the model forward pass never
    blocks the event loop, and tokens stream back via call_soon_threadsafe."""

    def __init__(self, model, tokenizer, engine_config: EngineConfig) -> None:
        self.tokenizer = tokenizer
        self.scheduler = Scheduler(model, engine_config)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._subscribers: Dict[int, _Subscriber] = {}
        self._ids = itertools.count()
        self._thread = threading.Thread(target=self._run, daemon=True, name="engine-worker")
        self._thread.start()

    def submit(
        self,
        prompt_ids: List[int],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: Optional[int],
        loop: asyncio.AbstractEventLoop,
    ) -> Tuple[int, "asyncio.Queue"]:
        request_id = next(self._ids)
        sub = _Subscriber(loop=loop, queue=asyncio.Queue())
        request = Request(
            request_id=request_id,
            prompt_ids=prompt_ids,
            max_new_tokens=max_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
        with self._lock:
            self.scheduler.add_request(request)
            self._subscribers[request_id] = sub
        self._wake.set()
        return request_id, sub.queue

    def abort(self, request_id: int) -> None:
        """Called when a client goes away; frees the slot immediately."""

        with self._lock:
            self.scheduler.abort_request(request_id)
            self._subscribers.pop(request_id, None)

    def _run(self) -> None:
        while True:
            self._wake.wait()
            with self._lock:
                if not self.scheduler.has_unfinished:
                    self._wake.clear()
                    continue
                out = self.scheduler.step()
                events: List[Tuple[_Subscriber, Any]] = []
                for rid, token in out.new_tokens.items():
                    sub = self._subscribers[rid]
                    sub.token_ids.append(token)
                    events.append((sub, token))
                for request in out.finished:
                    sub = self._subscribers.pop(request.request_id)
                    stopped = (
                        request.eos_token_id is not None
                        and request.generated_ids
                        and request.generated_ids[-1] == request.eos_token_id
                    )
                    events.append((sub, (_FINISHED, "stop" if stopped else "length")))
            for sub, event in events:
                sub.loop.call_soon_threadsafe(sub.queue.put_nowait, event)


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False
    seed: Optional[int] = None
    n: int = Field(default=1, ge=1, le=1)  # multiple choices per prompt not supported


class TokenStream:
    """Consumes one request's token queue and yields incremental text deltas.

    Decoding the full id list each time and slicing off the already-emitted
    prefix keeps multi-byte and merged tokens correct; per-token decode does not.
    """

    def __init__(self, worker: EngineWorker, request_id: int, queue: "asyncio.Queue") -> None:
        self.worker = worker
        self.request_id = request_id
        self.queue = queue
        self.token_ids: List[int] = []
        self._emitted = ""
        self.finish_reason: Optional[str] = None

    async def __aiter__(self) -> AsyncIterator[str]:
        while True:
            event = await self.queue.get()
            if isinstance(event, tuple) and event[0] is _FINISHED:
                self.finish_reason = event[1]
                return
            self.token_ids.append(event)
            text = self.worker.tokenizer.decode(self.token_ids, skip_special_tokens=True)
            delta, self._emitted = text[len(self._emitted):], text
            if delta:
                yield delta


def create_app(worker: EngineWorker, model_name: str = DEFAULT_MODEL_ID) -> FastAPI:
    app = FastAPI(title="mini-vllm")

    @app.get("/v1/models")
    async def list_models() -> JSONResponse:
        return JSONResponse(
            {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "mini-vllm"}]}
        )

    @app.post("/v1/completions")
    async def completions(body: CompletionRequest):
        prompts = [body.prompt] if isinstance(body.prompt, str) else list(body.prompt)
        if not prompts:
            raise HTTPException(status_code=400, detail="prompt must be non-empty")
        loop = asyncio.get_running_loop()
        streams: List[TokenStream] = []
        prompt_token_count = 0
        for prompt in prompts:
            prompt_ids = worker.tokenizer.encode(prompt)
            prompt_token_count += len(prompt_ids)
            try:
                request_id, queue = worker.submit(
                    prompt_ids,
                    max_tokens=body.max_tokens,
                    temperature=body.temperature,
                    top_p=body.top_p,
                    seed=body.seed,
                    loop=loop,
                )
            except ValueError as exc:  # prompt exceeds block pool or context limit
                for started in streams:
                    worker.abort(started.request_id)
                raise HTTPException(status_code=400, detail=str(exc))
            streams.append(TokenStream(worker, request_id, queue))

        completion_id = f"cmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if body.stream:
            return StreamingResponse(
                _sse_stream(completion_id, created, model_name, streams),
                media_type="text/event-stream",
            )

        choices = []
        completion_tokens = 0
        try:
            for index, stream in enumerate(streams):
                parts = [delta async for delta in stream]
                completion_tokens += len(stream.token_ids)
                choices.append(
                    {
                        "text": "".join(parts),
                        "index": index,
                        "logprobs": None,
                        "finish_reason": stream.finish_reason,
                    }
                )
        finally:
            # Handler cancellation (client gone) must not leave sequences running.
            for stream in streams:
                if stream.finish_reason is None:
                    worker.abort(stream.request_id)
        return JSONResponse(
            {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": model_name,
                "choices": choices,
                "usage": {
                    "prompt_tokens": prompt_token_count,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_token_count + completion_tokens,
                },
            }
        )

    return app


async def _sse_stream(
    completion_id: str, created: int, model_name: str, streams: List[TokenStream]
) -> AsyncIterator[str]:
    def chunk(index: int, text: str, finish_reason: Optional[str]) -> str:
        payload = {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "choices": [
                {"text": text, "index": index, "logprobs": None, "finish_reason": finish_reason}
            ],
        }
        return f"data: {json.dumps(payload)}\n\n"

    # Interleave choices as their tokens arrive so concurrent prompts stream live.
    async def pump(index: int, stream: TokenStream, out: "asyncio.Queue") -> None:
        async for delta in stream:
            await out.put(chunk(index, delta, None))
        await out.put(chunk(index, "", stream.finish_reason))
        await out.put(None)

    out: "asyncio.Queue[Optional[str]]" = asyncio.Queue()
    tasks = [asyncio.create_task(pump(i, s, out)) for i, s in enumerate(streams)]
    remaining = len(tasks)
    try:
        while remaining:
            item = await out.get()
            if item is None:
                remaining -= 1
                continue
            yield item
        yield "data: [DONE]\n\n"
    finally:
        for task in tasks:
            task.cancel()
        # A dropped SSE connection lands here; release engine slots immediately.
        for stream in streams:
            if stream.finish_reason is None:
                stream.worker.abort(stream.request_id)


def main() -> None:
    import argparse

    import uvicorn

    from engine.inference import MiniVLLMEngine

    parser = argparse.ArgumentParser(description="mini-vllm OpenAI-compatible server")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--num-blocks", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    args = parser.parse_args()

    engine = MiniVLLMEngine.from_pretrained(args.model, device=args.device, dtype=args.dtype)
    worker = EngineWorker(
        engine.model,
        engine.tokenizer,
        EngineConfig(num_blocks=args.num_blocks, max_num_seqs=args.max_num_seqs),
    )
    uvicorn.run(create_app(worker, model_name=args.model), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
