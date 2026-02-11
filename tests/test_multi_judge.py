from promptlab.application.evaluate_response import (
    IndividualJudgeResult,
    JudgeResult,
)


def test_single_judge_is_not_multi():
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


def test_multiple_judges_is_multi():
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
