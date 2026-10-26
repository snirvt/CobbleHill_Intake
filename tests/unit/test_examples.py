import json

from classifier.output.examples import dr_nurse_match_example, dr_validity_example

_REPORT_KEYS = [
    "sharepoint_folder", "web_url", "summary", "all_category_groups",
]


def test_dr_validity_example_has_report_shape_with_success_and_failure() -> None:
    example = dr_validity_example()

    assert list(example) == [*_REPORT_KEYS, "patients", "errors"]
    patients = example["patients"]
    assert isinstance(patients, list)
    assert [p["success"] for p in patients] == [True, False]
    assert patients[0]["verdict"] == "VALID"
    assert patients[1]["error"]
    assert example["summary"] == {
        "total": 2, "valid": 1, "invalid": 0, "unknown_age": 0, "failed": 1,
        "skipped_folders": 1, "errors": 2,
    }


def test_dr_nurse_match_example_has_report_shape_with_success_and_failure() -> None:
    example = dr_nurse_match_example()

    assert list(example) == [*_REPORT_KEYS, "pairs", "errors"]
    pairs = example["pairs"]
    assert isinstance(pairs, list)
    assert [p["success"] for p in pairs] == [True, False]
    assert pairs[0]["overall"] == "MATCH"
    assert pairs[0]["folder"].startswith("https://sharepoint.example/")
    assert example["summary"] == {
        "total": 2, "match": 1, "mismatch": 0, "undecided": 0, "failed": 1,
        "skipped_folders": 1, "errors": 2,
    }


def test_examples_are_obviously_fake_and_json_serialisable() -> None:
    for example in (dr_validity_example(), dr_nurse_match_example()):
        text = json.dumps(example)
        assert "EXAMPLE" in text
        assert "/tmp" not in text and "cobblehill_api_" not in text
