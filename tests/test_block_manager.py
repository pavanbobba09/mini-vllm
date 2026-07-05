import pytest

from engine.block_manager import BlockManager, OutOfBlocksError


def test_allocate_and_free_roundtrip():
    bm = BlockManager(num_blocks=8, block_size=16)
    bm.allocate_sequence(seq_id=1, num_tokens=40)  # 40 tokens -> 3 blocks
    assert bm.num_free_blocks == 5
    assert len(bm.block_table(1)) == 3
    bm.free_sequence(1)
    assert bm.num_free_blocks == 8


def test_double_free_raises():
    bm = BlockManager(num_blocks=4, block_size=16)
    bm.allocate_sequence(seq_id=1, num_tokens=10)
    bm.free_sequence(1)
    with pytest.raises(KeyError):
        bm.free_sequence(1)


def test_out_of_blocks_raises_and_leaves_state_clean():
    bm = BlockManager(num_blocks=2, block_size=16)
    with pytest.raises(OutOfBlocksError):
        bm.allocate_sequence(seq_id=1, num_tokens=33)  # needs 3 blocks
    # Failed allocation must not leak a partial table.
    assert bm.num_free_blocks == 2
    assert not bm.has_sequence(1)


def test_free_list_reuse():
    bm = BlockManager(num_blocks=4, block_size=16)
    first = bm.allocate_sequence(seq_id=1, num_tokens=32)
    bm.free_sequence(1)
    second = bm.allocate_sequence(seq_id=2, num_tokens=32)
    # LIFO free list: a new sequence reuses the blocks just freed.
    assert set(second) == set(first)


def test_append_token_grows_table_at_block_boundary():
    bm = BlockManager(num_blocks=4, block_size=4)
    bm.allocate_sequence(seq_id=1, num_tokens=4)  # exactly one full block
    assert len(bm.block_table(1)) == 1
    slot = bm.append_token(1)  # token 5 must open block 2
    assert len(bm.block_table(1)) == 2
    table = bm.block_table(1)
    assert slot == table[1] * 4 + 0
    assert bm.num_tokens(1) == 5


def test_append_token_out_of_blocks():
    bm = BlockManager(num_blocks=1, block_size=4)
    bm.allocate_sequence(seq_id=1, num_tokens=4)
    with pytest.raises(OutOfBlocksError):
        bm.append_token(1)
    # Failed growth must not corrupt the token count.
    assert bm.num_tokens(1) == 4


def test_slot_mapping_matches_block_table():
    bm = BlockManager(num_blocks=8, block_size=4)
    bm.allocate_sequence(seq_id=7, num_tokens=10)
    table = bm.block_table(7)
    slots = bm.slots_for_range(7, 0, 10)
    for pos, slot in enumerate(slots):
        assert slot == table[pos // 4] * 4 + pos % 4


def test_no_leak_over_many_sequences():
    bm = BlockManager(num_blocks=16, block_size=4)
    for i in range(100):
        n = (i % 40) + 1  # lengths 1..40, up to 10 blocks
        bm.allocate_sequence(seq_id=i, num_tokens=n)
        for _ in range(i % 3):
            bm.append_token(i)
        bm.free_sequence(i)
    assert bm.num_free_blocks == 16


def test_two_sequences_get_disjoint_blocks():
    bm = BlockManager(num_blocks=8, block_size=4)
    a = bm.allocate_sequence(seq_id=1, num_tokens=8)
    b = bm.allocate_sequence(seq_id=2, num_tokens=8)
    assert set(a).isdisjoint(set(b))
