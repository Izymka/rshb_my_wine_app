from pathlib import Path

from eval.platform_benchmark import Outcome, summarize
from wine_scanner.catalog import Query


def test_precision_includes_unknown_false_accepts_in_overall_metric():
    known = Query(Path("known.jpg"), "wine-a", "front", wine_id="wine-a")
    stranger = Query(Path("unknown.jpg"), "unknown:other", "hard", wine_id="other")
    rows = [
        Outcome(known, True, answered=True, verdict="correct", final_top1=True),
        Outcome(known, True, answered=False, verdict="refused", final_top1=True),
        Outcome(stranger, False, answered=True, verdict="accepted"),
    ]
    result = summarize(rows)
    assert result["answered_precision"] == 1.0
    assert result["overall_answered_precision"] == 0.5
    assert result["coverage"] == 0.5
    assert result["known_correct_coverage"] == 0.5
    assert result["final_top1"] == 1.0
    assert result["false_accept"]["all"]["rate"] == 1.0
