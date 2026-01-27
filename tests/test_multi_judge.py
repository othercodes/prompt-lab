"""Tests for multi-judge aggregation logic."""

import pytest

from promptlab.application.evaluate_response import (
    EvaluateResponse,
    IndividualJudgeResult,
    JudgeResult,
)


class TestAggregateScores:
    """Test score aggregation methods."""

    @pytest.fixture
    def evaluator(self):
        # Create evaluator with dummy provider factory
        return EvaluateResponse(provider_factory=lambda x: None)

    def test_mean_aggregation(self, evaluator):
        scores = [7, 8, 9]
        result = evaluator._aggregate_scores(scores, "mean")
        assert result == 8  # (7+8+9)/3 = 8

    def test_mean_aggregation_rounds(self, evaluator):
        scores = [7, 8]
        result = evaluator._aggregate_scores(scores, "mean")
        assert result == 8  # (7+8)/2 = 7.5 -> rounds to 8

    def test_median_aggregation_odd(self, evaluator):
        scores = [5, 8, 9]
        result = evaluator._aggregate_scores(scores, "median")
        assert result == 8  # middle value

    def test_median_aggregation_even(self, evaluator):
        scores = [5, 7, 8, 10]
        result = evaluator._aggregate_scores(scores, "median")
        assert result == 8  # (7+8)/2 = 7.5 -> rounds to 8

    def test_unknown_method_defaults_to_mean(self, evaluator):
        scores = [6, 8, 10]
        result = evaluator._aggregate_scores(scores, "unknown")
        assert result == 8  # falls back to mean


class TestJudgeResult:
    """Test JudgeResult dataclass."""

    def test_single_judge_not_multi(self):
        result = JudgeResult(
            score=8,
            reasoning="Good response",
            raw={"score": 8},
            individual_results=[
                IndividualJudgeResult(
                    model="openai:gpt-4o",
                    score=8,
                    reasoning="Good",
                    raw={"score": 8},
                )
            ],
        )
        assert result.is_multi_judge is False

    def test_multi_judge_detected(self):
        result = JudgeResult(
            score=8,
            reasoning="Combined reasoning",
            raw={"aggregation": "mean"},
            individual_results=[
                IndividualJudgeResult(
                    model="openai:gpt-4o-mini",
                    score=7,
                    reasoning="Okay",
                    raw={"score": 7},
                ),
                IndividualJudgeResult(
                    model="anthropic:claude-sonnet-4-20250514",
                    score=9,
                    reasoning="Great",
                    raw={"score": 9},
                ),
            ],
        )
        assert result.is_multi_judge is True


class TestIndividualJudgeResult:
    """Test IndividualJudgeResult dataclass."""

    def test_fields(self):
        result = IndividualJudgeResult(
            model="openai:gpt-4o",
            score=8,
            reasoning="Good response",
            raw={"score": 8, "reasoning": "Good response"},
        )
        assert result.model == "openai:gpt-4o"
        assert result.score == 8
        assert result.reasoning == "Good response"
        assert result.raw == {"score": 8, "reasoning": "Good response"}
