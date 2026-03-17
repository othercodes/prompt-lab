from io import StringIO
from unittest.mock import patch

from rich.console import Console

from promptlab.domain.contracts.results import RunResult, RunSummary
from promptlab.infrastructure.console_display import (
    _display_individual_results_table,
    _score_style,
)


# --- _score_style ---


def test_score_style_default_range_green():
    # range_size=9; green >= 1+7.2=8.2, so 9 and 10 are green
    assert _score_style(9, (1, 10)) == "bold green"
    assert _score_style(10, (1, 10)) == "bold green"


def test_score_style_default_range_yellow():
    # yellow >= 1+5.4=6.4, so 7 and 8 are yellow
    assert _score_style(7, (1, 10)) == "yellow"
    assert _score_style(8, (1, 10)) == "yellow"


def test_score_style_default_range_orange():
    # orange >= 1+3.6=4.6, so 5 and 6 are orange
    assert _score_style(5, (1, 10)) == "orange3"
    assert _score_style(6, (1, 10)) == "orange3"


def test_score_style_default_range_red():
    # below 4.6, so 1-4 are red
    assert _score_style(1, (1, 10)) == "bold red"
    assert _score_style(4, (1, 10)) == "bold red"


def test_score_style_range_1_5():
    # range_size = 4; thresholds: green >= 1+3.2=4.2, yellow >= 1+2.4=3.4, orange >= 1+1.6=2.6
    assert _score_style(5, (1, 5)) == "bold green"
    assert _score_style(4, (1, 5)) == "yellow"
    assert _score_style(3, (1, 5)) == "orange3"
    assert _score_style(1, (1, 5)) == "bold red"


def test_score_style_range_1_20():
    # range_size = 19; thresholds: green >= 1+15.2=16.2, yellow >= 1+11.4=12.4, orange >= 1+7.6=8.6
    assert _score_style(17, (1, 20)) == "bold green"
    assert _score_style(13, (1, 20)) == "yellow"
    assert _score_style(9, (1, 20)) == "orange3"
    assert _score_style(5, (1, 20)) == "bold red"


# --- _display_individual_results_table ---


def _make_result(score: int) -> RunResult:
    return RunResult(
        input_id="test-input",
        model="test:model",
        run_number=1,
        cached=False,
        latency_ms=100,
        input_tokens=10,
        output_tokens=20,
        response={"content": "test"},
        judge={"score": score},
    )


def _make_summary(score_range: tuple[int, int], results: list[RunResult]) -> RunSummary:
    return RunSummary(
        timestamp="20240101_120000",
        experiment="test-exp",
        variant="v1",
        models=["test:model"],
        inputs_count=len(results),
        runs_per_input=1,
        duration_seconds=1.0,
        cached_responses=0,
        score_range=score_range,
        results=results,
    )


def test_individual_results_table_shows_correct_score_max():
    buf = StringIO()
    test_console = Console(file=buf, highlight=False, markup=False)

    summary = _make_summary((1, 5), [_make_result(4)])

    with patch("promptlab.infrastructure.console_display.console", test_console):
        _display_individual_results_table(summary, show_hypothesis=False)

    output = buf.getvalue()
    assert "/5" in output
    assert "/10" not in output


def test_individual_results_table_default_range_shows_10():
    buf = StringIO()
    test_console = Console(file=buf, highlight=False, markup=False)

    summary = _make_summary((1, 10), [_make_result(8)])

    with patch("promptlab.infrastructure.console_display.console", test_console):
        _display_individual_results_table(summary, show_hypothesis=False)

    output = buf.getvalue()
    assert "/10" in output
