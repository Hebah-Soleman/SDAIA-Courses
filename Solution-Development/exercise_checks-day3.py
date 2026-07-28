import math


def _ok(msg: str):
    print(f"✅ {msg}")


# ---------------------------------------------------------------------------
# Exercise 1 — SlidingWindowChunker (word-based, not character-based)
# ---------------------------------------------------------------------------
def check_exercise_1(chunker_cls):
    chunker = chunker_cls(window_size=5, stride=3)
    text = "one two three four five six seven eight nine ten"
    chunks = chunker.chunk(text)

    assert isinstance(chunks, list) and len(chunks) > 0, (
        "chunk() should return a non-empty list of strings."
    )
    assert all(isinstance(c, str) for c in chunks), "Every chunk should be a string."

    # window_size=5, stride=3, 10 words -> windows start at 0,3,6,9
    expected = [
        "one two three four five",
        "four five six seven eight",
        "seven eight nine ten",
        "ten",
    ]
    assert chunks == expected, (
        f"With window_size=5, stride=3 on a 10-word input, expected windows "
        f"starting at word indices 0,3,6,9:\n{expected}\ngot:\n{chunks}\n"
        "Remember stride controls how many words you advance the window each step, "
        "and the last window should just take whatever words remain."
    )

    # No overlap when stride >= window_size
    chunker2 = chunker_cls(window_size=4, stride=4)
    chunks2 = chunker2.chunk("a b c d e f g h")
    assert chunks2 == ["a b c d", "e f g h"], (
        f"With window_size=4, stride=4 (no overlap), expected ['a b c d', 'e f g h'], got {chunks2}."
    )

    _ok("Exercise 1 passed — SlidingWindowChunker produces correctly overlapping word windows.")


# ---------------------------------------------------------------------------
# Exercise 2 — chunk_stats
# ---------------------------------------------------------------------------
def check_exercise_2(chunk_stats):
    chunks = ["ab", "abcd", "abcdef", "abcdefgh"]  # lengths 2, 4, 6, 8
    stats = chunk_stats(chunks)

    assert isinstance(stats, dict), "chunk_stats should return a dict."
    for key in ("count", "min_len", "max_len", "avg_len", "std_len"):
        assert key in stats, f"chunk_stats result is missing key '{key}'."

    assert stats["count"] == 4, f"count should be 4, got {stats['count']}."
    assert stats["min_len"] == 2, f"min_len should be 2, got {stats['min_len']}."
    assert stats["max_len"] == 8, f"max_len should be 8, got {stats['max_len']}."
    assert abs(stats["avg_len"] - 5.0) < 1e-9, f"avg_len should be 5.0, got {stats['avg_len']}."

    expected_std = math.sqrt(sum((x - 5.0) ** 2 for x in (2, 4, 6, 8)) / 4)
    assert abs(stats["std_len"] - expected_std) < 1e-6, (
        f"std_len should be the population standard deviation ({expected_std:.4f}), "
        f"got {stats['std_len']}."
    )

    # Edge case: empty input shouldn't crash
    empty_stats = chunk_stats([])
    assert empty_stats["count"] == 0, "chunk_stats([]) should report count == 0."

    _ok("Exercise 2 passed — chunk_stats correctly summarizes chunk length distribution.")


# ---------------------------------------------------------------------------
# Exercise 3 — tiered_embedding_cost
# ---------------------------------------------------------------------------
def check_exercise_3(tiered_embedding_cost):
    # Tiers: first 1M tokens at $0.02/1M, everything above that at $0.01/1M
    # 500,000 tokens -> entirely in first tier
    cost_small = tiered_embedding_cost(500_000)
    expected_small = (500_000 / 1_000_000) * 0.02
    assert abs(cost_small - expected_small) < 1e-9, (
        f"For 500,000 tokens (under the 1M threshold), expected cost ${expected_small:.4f}, "
        f"got {cost_small}."
    )

    # 1,500,000 tokens -> 1M at tier-1 rate + 500K at tier-2 rate
    cost_large = tiered_embedding_cost(1_500_000)
    expected_large = (1_000_000 / 1_000_000) * 0.02 + (500_000 / 1_000_000) * 0.01
    assert abs(cost_large - expected_large) < 1e-9, (
        f"For 1,500,000 tokens, expected the first 1M billed at $0.02/1M and the "
        f"remaining 500K billed at $0.01/1M, total ${expected_large:.4f}, got {cost_large}. "
        "Don't apply the discounted rate to the whole amount — only the portion over 1M tokens."
    )

    _ok("Exercise 3 passed — tiered_embedding_cost correctly applies the discount only above the threshold.")


# ---------------------------------------------------------------------------
# Exercise 4 — precision_at_k and average_precision
# ---------------------------------------------------------------------------
def check_exercise_4(precision_at_k, average_precision):
    retrieved = ["d3", "d1", "d7", "d2", "d9"]
    relevant = {"d1", "d2"}

    # top-3 = [d3, d1, d7] -> 1 of 3 relevant
    p3 = precision_at_k(retrieved, relevant, 3)
    assert abs(p3 - (1 / 3)) < 1e-9, f"precision_at_3 should be 1/3 ≈ 0.333, got {p3}."

    # top-5 = all 5 -> 2 of 5 relevant
    p5 = precision_at_k(retrieved, relevant, 5)
    assert abs(p5 - 0.4) < 1e-9, f"precision_at_5 should be 0.4, got {p5}."

    # Average precision: average of precision@k at each rank where a relevant doc appears.
    # Relevant hits are at rank 2 (d1) -> precision@2 = 1/2, and rank 4 (d2) -> precision@4 = 2/4
    ap = average_precision(retrieved, relevant)
    expected_ap = ((1 / 2) + (2 / 4)) / 2
    assert abs(ap - expected_ap) < 1e-9, (
        f"average_precision should be {expected_ap:.4f} "
        f"(mean of precision@rank at each relevant hit: precision@2=0.5, precision@4=0.5), got {ap}."
    )

    # Edge case: no relevant docs retrieved at all -> AP is 0.0
    ap_zero = average_precision(["x1", "x2"], {"d1"})
    assert ap_zero == 0.0, "average_precision should be 0.0 when no relevant docs are retrieved."

    _ok("Exercise 4 passed — precision_at_k and average_precision are computed correctly.")


# ---------------------------------------------------------------------------
# Exercise 5 — cosine_similarity + most_similar_pair (no numpy)
# ---------------------------------------------------------------------------
def check_exercise_5(cosine_similarity, most_similar_pair):
    v1 = [1.0, 0.0]
    v2 = [0.0, 1.0]
    v3 = [1.0, 0.0]

    sim_orthogonal = cosine_similarity(v1, v2)
    assert abs(sim_orthogonal - 0.0) < 1e-9, f"cosine_similarity of orthogonal vectors should be 0.0, got {sim_orthogonal}."

    sim_identical = cosine_similarity(v1, v3)
    assert abs(sim_identical - 1.0) < 1e-9, f"cosine_similarity of identical-direction vectors should be 1.0, got {sim_identical}."

    v4 = [3.0, 4.0]
    v5 = [-3.0, -4.0]
    sim_opposite = cosine_similarity(v4, v5)
    assert abs(sim_opposite - (-1.0)) < 1e-9, f"cosine_similarity of opposite vectors should be -1.0, got {sim_opposite}."

    vectors = {
        "a": [1.0, 0.0],
        "b": [0.99, 0.01],
        "c": [0.0, 1.0],
    }
    pair, score = most_similar_pair(vectors)
    assert set(pair) == {"a", "b"}, (
        f"most_similar_pair should find 'a' and 'b' as the closest pair (nearly identical direction), got {pair}."
    )
    assert score > 0.99, f"Similarity for the most similar pair should be > 0.99, got {score}."

    _ok("Exercise 5 passed — cosine_similarity and most_similar_pair work correctly without numpy.")
