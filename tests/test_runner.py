from promptlab.runner import (
    RunResult,
    calculate_confidence_interval,
    calculate_stats,
    welch_t_test,
)


def test_calculate_stats_single_run():
    results = [
        RunResult(
            input_id="test-1",
            model="openai:gpt-4o",
            run_number=1,
            cached=False,
            latency_ms=100,
            input_tokens=10,
            output_tokens=20,
            response={"content": "Hello"},
            judge={"score": 8, "reasoning": "Good"},
        ),
    ]

    stats = calculate_stats(results)

    assert len(stats) == 1
    assert stats[0].input_id == "test-1"
    assert stats[0].model == "openai:gpt-4o"
    assert stats[0].runs == 1
    assert stats[0].mean == 8.0
    assert stats[0].stddev == 0.0
    assert stats[0].min_score == 8
    assert stats[0].max_score == 8
    assert stats[0].scores == [8]
    assert stats[0].ci_lower == 8.0
    assert stats[0].ci_upper == 8.0


def test_calculate_stats_multiple_runs():
    results = [
        RunResult(
            input_id="test-1",
            model="openai:gpt-4o",
            run_number=1,
            cached=False,
            latency_ms=100,
            input_tokens=10,
            output_tokens=20,
            response={"content": "Hello"},
            judge={"score": 8, "reasoning": "Good"},
        ),
        RunResult(
            input_id="test-1",
            model="openai:gpt-4o",
            run_number=2,
            cached=False,
            latency_ms=110,
            input_tokens=10,
            output_tokens=22,
            response={"content": "Hi"},
            judge={"score": 10, "reasoning": "Great"},
        ),
        RunResult(
            input_id="test-1",
            model="openai:gpt-4o",
            run_number=3,
            cached=False,
            latency_ms=105,
            input_tokens=10,
            output_tokens=21,
            response={"content": "Hey"},
            judge={"score": 9, "reasoning": "Very good"},
        ),
    ]

    stats = calculate_stats(results)

    assert len(stats) == 1
    assert stats[0].input_id == "test-1"
    assert stats[0].runs == 3
    assert stats[0].mean == 9.0
    assert stats[0].stddev == 1.0
    assert stats[0].min_score == 8
    assert stats[0].max_score == 10
    assert stats[0].scores == [8, 10, 9]
    # CI should be calculated (mean=9, stddev=1, n=3, t_crit~=4.303)
    assert stats[0].ci_lower < stats[0].mean
    assert stats[0].ci_upper > stats[0].mean


def test_calculate_stats_groups_by_input_and_model():
    results = [
        RunResult(
            input_id="test-1",
            model="openai:gpt-4o",
            run_number=1,
            cached=False,
            latency_ms=100,
            input_tokens=10,
            output_tokens=20,
            response={"content": "A"},
            judge={"score": 8, "reasoning": "Good"},
        ),
        RunResult(
            input_id="test-1",
            model="anthropic:claude-sonnet",
            run_number=1,
            cached=False,
            latency_ms=150,
            input_tokens=10,
            output_tokens=25,
            response={"content": "B"},
            judge={"score": 9, "reasoning": "Great"},
        ),
        RunResult(
            input_id="test-2",
            model="openai:gpt-4o",
            run_number=1,
            cached=False,
            latency_ms=100,
            input_tokens=10,
            output_tokens=20,
            response={"content": "C"},
            judge={"score": 7, "reasoning": "OK"},
        ),
    ]

    stats = calculate_stats(results)

    assert len(stats) == 3
    stats_dict = {(s.input_id, s.model): s for s in stats}

    assert ("test-1", "openai:gpt-4o") in stats_dict
    assert ("test-1", "anthropic:claude-sonnet") in stats_dict
    assert ("test-2", "openai:gpt-4o") in stats_dict

    assert stats_dict[("test-1", "openai:gpt-4o")].mean == 8.0
    assert stats_dict[("test-1", "anthropic:claude-sonnet")].mean == 9.0
    assert stats_dict[("test-2", "openai:gpt-4o")].mean == 7.0


def test_confidence_interval_single_sample():
    ci_lower, ci_upper = calculate_confidence_interval(8.0, 0.0, 1)
    assert ci_lower == 8.0
    assert ci_upper == 8.0


def test_confidence_interval_multiple_samples():
    # mean=9, stddev=1, n=3
    # t_crit for df=2 is 4.303
    # margin = 4.303 * (1 / sqrt(3)) = 4.303 * 0.577 = 2.48
    ci_lower, ci_upper = calculate_confidence_interval(9.0, 1.0, 3)
    assert ci_lower < 9.0
    assert ci_upper > 9.0
    assert 6.0 < ci_lower < 7.0  # ~6.52
    assert 11.0 < ci_upper < 12.0  # ~11.48


def test_welch_t_test_same_distributions():
    scores1 = [8, 9, 8, 9, 8]
    scores2 = [8, 9, 8, 9, 8]
    t_stat, p_value = welch_t_test(scores1, scores2)
    assert t_stat == 0.0
    assert p_value == 1.0


def test_welch_t_test_different_distributions():
    scores1 = [9, 10, 9, 10, 9]  # mean ~9.4
    scores2 = [5, 6, 5, 6, 5]  # mean ~5.4
    t_stat, p_value = welch_t_test(scores1, scores2)
    assert t_stat > 0  # scores1 > scores2
    assert p_value <= 0.05  # Should be significant


def test_welch_t_test_insufficient_samples():
    scores1 = [8]
    scores2 = [9]
    t_stat, p_value = welch_t_test(scores1, scores2)
    assert t_stat == 0.0
    assert p_value == 1.0
