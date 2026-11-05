import json

from classifier.output.examples import (
    dr_nurse_match_example,
    dr_validity_example,
    intake_review_example,
)

_REPORT_KEYS = [
    "sharepoint_folder", "web_url", "summary", "retryable", "all_category_groups",
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


def test_dr_validity_example_shows_mismatch_warning_and_uncategorized_diagnosis() -> None:
    patient = dr_validity_example()["patients"][0]  # type: ignore[index]

    assert len(patient["warnings"]) == 1
    assert patient["warnings"][0].startswith("Category count mismatch after ")
    assert [d["category"] for d in patient["diagnoses"]] == ["Other", None]


def test_dr_nurse_match_example_shows_nurse_note_mismatch_warning() -> None:
    pair = dr_nurse_match_example()["pairs"][0]  # type: ignore[index]

    assert len(pair["warnings"]) == 1
    assert pair["warnings"][0].startswith("nurse note: Category count mismatch after ")
    assert pair["nurse_diagnoses"][-1]["category"] is None


def test_intake_review_example_accepted_patient_shows_nurse_warning() -> None:
    patients = intake_review_example()["patients"]
    accepted = patients[0]  # type: ignore[index]

    assert accepted["final_verdict"] == "ACCEPT"
    assert [w.split(":")[0] for w in accepted["warnings"]] == ["nurse note"]
    others = [p["warnings"] for p in patients[1:] if "warnings" in p]  # type: ignore[index]
    assert not any(w.startswith("nurse note:") for ws in others for w in (ws or []))


def test_examples_show_retryable_flags() -> None:
    for example in (dr_validity_example(), dr_nurse_match_example(), intake_review_example()):
        assert example["retryable"] is True
        errors = example["errors"]
        assert isinstance(errors, list)
        for error in errors:
            assert error["retryable"] is (error["code"] in ("llm_failed", "parse_failed"))
        assert {e["retryable"] for e in errors} == {True, False}
