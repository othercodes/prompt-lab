import math
import statistics as stats
from dataclasses import dataclass

from .contracts.results import InputStats, RunResult, RunSummary

# t-critical values for 95% CI (two-tailed, alpha=0.05)
# df -> t_critical
_T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    15: 2.131,
    20: 2.086,
    30: 2.042,
    50: 2.009,
    100: 1.984,
}

# p-value thresholds for t-distribution (approximate, two-tailed)
# Format: {df: [(t_value, p_value), ...]}
_P_VALUE_TABLE = {
    2: [(4.303, 0.05), (6.965, 0.01)],
    3: [(3.182, 0.05), (4.541, 0.01)],
    4: [(2.776, 0.05), (3.747, 0.01)],
    5: [(2.571, 0.05), (3.365, 0.01)],
    6: [(2.447, 0.05), (3.143, 0.01)],
    7: [(2.365, 0.05), (2.998, 0.01)],
    8: [(2.306, 0.05), (2.896, 0.01)],
    9: [(2.262, 0.05), (2.821, 0.01)],
    10: [(2.228, 0.05), (2.764, 0.01)],
    15: [(2.131, 0.05), (2.602, 0.01)],
    20: [(2.086, 0.05), (2.528, 0.01)],
    30: [(2.042, 0.05), (2.457, 0.01)],
    50: [(2.009, 0.05), (2.403, 0.01)],
    100: [(1.984, 0.05), (2.364, 0.01)],
}


def _get_t_critical(df: int) -> float:
    if df in _T_CRITICAL_95:
        return _T_CRITICAL_95[df]
    for threshold in sorted(_T_CRITICAL_95.keys(), reverse=True):
        if df >= threshold:
            return _T_CRITICAL_95[threshold]
    return 1.96  # Fallback to z-value for large samples


def _get_p_value_approx(t_stat: float, df: int) -> float:
    t_abs = abs(t_stat)
    closest_df = min(_P_VALUE_TABLE.keys(), key=lambda x: abs(x - df))
    thresholds = _P_VALUE_TABLE[closest_df]

    for t_thresh, p_val in thresholds:
        if t_abs >= t_thresh:
            return p_val

    return 1.0  # Not significant


def calculate_confidence_interval(
    mean: float, stddev: float, n: int
) -> tuple[float, float]:
    if n < 2:
        return mean, mean

    df = n - 1
    t_crit = _get_t_critical(df)
    margin = t_crit * (stddev / math.sqrt(n))

    return round(mean - margin, 2), round(mean + margin, 2)


def welch_t_test(scores1: list[int], scores2: list[int]) -> tuple[float, float]:
    n1, n2 = len(scores1), len(scores2)

    if n1 < 2 or n2 < 2:
        return 0.0, 1.0

    mean1 = stats.mean(scores1)
    mean2 = stats.mean(scores2)
    var1 = stats.variance(scores1)
    var2 = stats.variance(scores2)

    se = math.sqrt(var1 / n1 + var2 / n2)
    if se == 0:
        # Zero variance in both groups - if means differ, difference is deterministic
        if mean1 != mean2:
            return float("inf") if mean1 > mean2 else float("-inf"), 0.01
        return 0.0, 1.0

    t_stat = (mean1 - mean2) / se

    # Welch-Satterthwaite degrees of freedom
    num = (var1 / n1 + var2 / n2) ** 2
    denom = (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
    if denom == 0:
        df = float(n1 + n2 - 2)
    else:
        df = num / denom

    p_value = _get_p_value_approx(t_stat, int(df))

    return round(t_stat, 3), p_value


@dataclass
class SignificanceResult:
    variant1: str
    variant2: str
    mean1: float
    mean2: float
    t_statistic: float
    p_value: float
    significant: bool
    winner: str | None


def compare_variants_significance(
    summaries: list[tuple[str, RunSummary]],
) -> list[SignificanceResult]:
    results = []

    for i, (name1, sum1) in enumerate(summaries):
        for name2, sum2 in summaries[i + 1 :]:
            scores1 = [r.judge["score"] for r in sum1.results]
            scores2 = [r.judge["score"] for r in sum2.results]

            if not scores1 or not scores2:
                continue

            mean1 = stats.mean(scores1)
            mean2 = stats.mean(scores2)

            t_stat, p_value = welch_t_test(scores1, scores2)
            significant = p_value <= 0.05

            winner = None
            if significant:
                winner = name1 if mean1 > mean2 else name2

            results.append(
                SignificanceResult(
                    variant1=name1,
                    variant2=name2,
                    mean1=round(mean1, 2),
                    mean2=round(mean2, 2),
                    t_statistic=t_stat,
                    p_value=p_value,
                    significant=significant,
                    winner=winner,
                )
            )

    return results


def calculate_stats(results: list[RunResult]) -> list[InputStats]:
    grouped: dict[tuple[str, str], list[RunResult]] = {}
    for result in results:
        key = (result.input_id, result.model)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(result)

    stats_list = []
    for (input_id, model), group in grouped.items():
        scores = [r.judge["score"] for r in group]
        n = len(scores)

        if n == 1:
            mean = float(scores[0])
            stddev = 0.0
        else:
            mean = stats.mean(scores)
            stddev = stats.stdev(scores)

        ci_lower, ci_upper = calculate_confidence_interval(mean, stddev, n)

        stats_list.append(
            InputStats(
                input_id=input_id,
                model=model,
                runs=n,
                scores=scores,
                mean=round(mean, 2),
                stddev=round(stddev, 2),
                min_score=min(scores),
                max_score=max(scores),
                ci_lower=ci_lower,
                ci_upper=ci_upper,
            )
        )

    return stats_list
