"""Unit tests for evaluation telemetry helpers."""

from thinker.evaluation import build_evaluation_series


def test_build_evaluation_series_empty():
    """Empty sample list should return structured empty payload."""
    series = build_evaluation_series([])
    assert series["indices"] == []
    assert series["timestamps"] == []
    assert series["cumulative_rates"] == {}
    assert series["value_series"] == {}
    assert series["moving_averages"] == {}


def test_build_evaluation_series_populates_cumulative_and_moving_average():
    """Verify cumulative rates + moving averages behave as expected."""
    samples = [
        {
            "index": 1,
            "timestamp": "2024-01-01T00:00:00Z",
            "schema_valid": True,
            "citation_valid": True,
            "entailment_score": 0.9,
            "entailment_pass": True,
            "semantic_similarity": 0.8,
            "similarity_pass": True,
            "paraphrase_accepted": False,
            "overall_pass": True,
            "beta1": 0.0,
            "chirality_score": 0.2,
            "fisher_rao_distance": 10.0,
        },
        {
            "index": 2,
            "timestamp": "2024-01-01T00:01:00Z",
            "schema_valid": False,
            "citation_valid": True,
            "entailment_score": 0.4,
            "entailment_pass": False,
            "semantic_similarity": 0.5,
            "similarity_pass": False,
            "paraphrase_accepted": False,
            "overall_pass": False,
            "beta1": 1.0,
            "chirality_score": 0.6,
            "fisher_rao_distance": 15.0,
        },
        {
            "index": 3,
            "timestamp": "2024-01-01T00:02:00Z",
            "schema_valid": True,
            "citation_valid": False,
            "entailment_score": 0.7,
            "entailment_pass": False,
            "semantic_similarity": 0.6,
            "similarity_pass": True,
            "paraphrase_accepted": True,
            "overall_pass": False,
            "beta1": 0.0,
            "chirality_score": 0.55,
            "fisher_rao_distance": 12.5,
        },
    ]
    series = build_evaluation_series(samples, moving_avg_window=2)

    assert series["indices"] == [1, 2, 3]
    assert series["timestamps"][0] == "2024-01-01T00:00:00Z"

    schema_rates = series["cumulative_rates"]["schema_valid"]
    assert schema_rates == [1.0, 0.5, 2 / 3]

    citation_series = series["value_series"]["citation_valid"]
    assert citation_series == [1, 1, 0]

    moving_entailment = series["moving_averages"]["entailment_score"]
    # moving average window=2 => first point uses just itself
    assert moving_entailment[0] == 0.9
    assert moving_entailment[1] == (0.9 + 0.4) / 2
    assert moving_entailment[2] == (0.4 + 0.7) / 2
