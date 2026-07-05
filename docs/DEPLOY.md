# Deploying mini-vllm for free (Hugging Face Spaces)

Hugging Face Spaces hosts Docker containers on a free CPU tier (2 vCPU, 16 GB
RAM) with a public URL. The 0.5B model runs comfortably there in fp32 on CPU.

Free-tier realities:

- The Space **sleeps** after about 48 hours without traffic and cold-starts on
  the next request. A cold start rebuilds nothing but re-downloads the ~1 GB
  model, so the first request after a sleep takes a minute or two.
- CPU inference runs roughly 15-30 tokens/s. Fine for a demo, not for load.

## 1. Create the Space

On huggingface.co: **New Space** -> name `mini-vllm` -> SDK: **Docker** ->
blank template -> hardware: **CPU basic (free)**.

## 2. Push this repo to the Space

Spaces read their config from a YAML block at the top of the README in the
Space repo. To keep the GitHub README clean, maintain a tiny `space` branch
that only adds that block:

```bash
git checkout -b space
```

Prepend this to `README.md` (including both `---` lines):

```yaml
---
title: mini-vllm
emoji: "⚡"
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
---
```

Then:

```bash
git commit -am "space: add Hugging Face frontmatter"
git remote add space https://huggingface.co/spaces/<hf-username>/mini-vllm
git push space space:main
git checkout main
```

Pushing to huggingface.co over https authenticates with a **write** access
token from hf.co/settings/tokens (used as the password).

To ship later code changes: `git checkout space && git merge main && git push space space:main`.

## 3. Set the API key

Space page -> **Settings** -> **Variables and secrets** -> add secret:

- name: `MINI_VLLM_API_KEY`
- value: any key you choose, e.g. the output of `openssl rand -hex 24`

Without the secret the server runs open; with it, every `/v1/*` request must
carry the key. `GET /health` always stays open so the platform can probe it.

## 4. First boot

The Space builds the Docker image, then the container downloads the model on
startup. Watch **Logs** on the Space page; ready when uvicorn prints its
startup line. The live URL is:

```
https://<hf-username>-mini-vllm.hf.space
```

## 5. Use it

```bash
curl https://<hf-username>-mini-vllm.hf.space/health
```

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://<hf-username>-mini-vllm.hf.space/v1",
    api_key="<your MINI_VLLM_API_KEY>",
)
print(client.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    prompt="The capital of France is",
    max_tokens=16,
).choices[0].text)
```

Streaming works the same way with `stream=True`.

## Running the image anywhere else

The same Dockerfile runs on any host with 2+ GB RAM (Render, Railway, a VPS):

```bash
docker build -t mini-vllm .
docker run -e MINI_VLLM_API_KEY=sk-demo -p 7860:7860 mini-vllm
```
