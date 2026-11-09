import json
from pathlib import Path

import pytest

from sourcetracker.cli import main

LABELS = Path(__file__).parent.parent / "examples" / "labels.json"


@pytest.fixture
def database(tmp_path, example_corpus, capsys):
    path = tmp_path / "index.db"
    assert main(["build", str(example_corpus), "--embedder", "hashing", "--db", str(path)]) == 0
    capsys.readouterr()
    return str(path)


def run(capsys, *args):
    code = main(list(args))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_build_reports_what_it_indexed(tmp_path, example_corpus, capsys):
    path = tmp_path / "index.db"
    code, out, _ = run(capsys, "build", str(example_corpus), "--embedder", "hashing", "--db", str(path))
    assert code == 0
    assert "Indexed 9 articles" in out


def test_build_accepts_threshold_overrides(tmp_path, example_corpus, capsys):
    path = str(tmp_path / "index.db")
    args = ["build", str(example_corpus), "--embedder", "hashing", "--db", path]
    run(capsys, *args, "--match-threshold", "1.5")
    _, out, _ = run(capsys, "audit", "--db", path)
    assert "0 matched" in out


def test_audit_finds_each_problem_in_the_example(database, capsys):
    code, out, _ = run(capsys, "audit", "--db", database)
    assert code == 0
    assert "coastal-blog -> city-budget: Property values" in out
    assert "newsletter-a <-> newsletter-b" in out
    assert "harbor-survey: 1 conclusions lost" in out
    assert "coastal-blog resembles harbor-survey" in out
    assert "harbor-survey resembles coastal-blog" not in out


def test_remove_traces_a_lost_conclusion_through_the_chain(database, capsys):
    code, out, _ = run(capsys, "remove", "harbor-survey", "--db", database)
    assert code == 0
    assert "Lost all cited support (1):" in out
    assert "policy-brief: The northern reef needs urchin removal" in out
    assert "relied on tribune-report" in out
    assert "which relied on harbor-survey" in out
    assert "Weakened (1):" in out


def test_remove_accepts_a_url(database, capsys):
    _, by_url, _ = run(capsys, "remove", "https://marlowbay.example/reports/reef-survey", "--db", database)
    _, by_id, _ = run(capsys, "remove", "harbor-survey", "--db", database)
    assert by_url == by_id


def test_remove_accepts_an_external_source(database, capsys):
    code, out, _ = run(capsys, "remove", "https://fisheries.example/landings", "--db", database)
    assert code == 0
    assert "Claims no longer supported: 1" in out


def test_remove_rejects_an_unknown_source(database, capsys):
    code, _, err = run(capsys, "remove", "harbour-survey", "--db", database)
    assert code == 2
    assert "Unknown source" in err


def test_search_marks_repetition_as_uncited(database, capsys):
    code, out, _ = run(capsys, "search", "urchin density northern reef animals per square meter", "--db", database)
    assert code == 0
    assert "cited by   tribune-report" in out
    assert "resembles  coastal-blog" in out
    assert "(no citation)" in out


def test_trace_maps_both_directions(database, capsys):
    code, out, _ = run(capsys, "trace", "tribune-report", "--db", database)
    assert code == 0
    assert "harbor-survey (2 matched)" in out
    assert "policy-brief (2 matched)" in out
    assert "All sources upstream: harbor-survey, otter-study" in out


def test_trace_rejects_an_unknown_article(database, capsys):
    code, _, err = run(capsys, "trace", "nope", "--db", database)
    assert code == 2
    assert "Unknown article" in err


def test_evaluate_prints_each_metric(database, capsys):
    code, out, _ = run(capsys, "evaluate", str(LABELS), "--db", database)
    assert code == 0
    for heading in ("Citation links", "Passage matches", "Unsupported-citation flags", "Uncited repetition"):
        assert heading in out


def test_evaluate_reports_invalid_json(database, tmp_path, capsys):
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    code, _, err = run(capsys, "evaluate", str(broken), "--db", database)
    assert code == 2
    assert "not valid JSON" in err


def test_label_template_writes_a_blank_file(database, tmp_path, capsys):
    output = tmp_path / "labels.json"
    code, _, _ = run(capsys, "label-template", "tribune-report", "-o", str(output), "--db", database)
    assert code == 0
    entries = json.loads(output.read_text(encoding="utf-8"))["articles"]["tribune-report"]["citations"]
    assert len(entries) == 3


def test_label_template_protects_existing_labels(database, tmp_path, capsys):
    output = tmp_path / "labels.json"
    output.write_text("hand-written work", encoding="utf-8")
    code, _, err = run(capsys, "label-template", "tribune-report", "-o", str(output), "--db", database)
    assert code == 2
    assert "already exists" in err
    assert output.read_text(encoding="utf-8") == "hand-written work"


def test_label_template_force_replaces(database, tmp_path, capsys):
    output = tmp_path / "labels.json"
    output.write_text("old", encoding="utf-8")
    code, _, _ = run(capsys, "label-template", "tribune-report", "-o", str(output), "--force", "--db", database)
    assert code == 0
    assert "citations" in output.read_text(encoding="utf-8")


def test_missing_index_is_a_clean_error(tmp_path, capsys):
    code, _, err = run(capsys, "audit", "--db", str(tmp_path / "missing.db"))
    assert code == 2
    assert "sourcetracker build" in err


def test_missing_corpus_is_a_clean_error(tmp_path, capsys):
    code, _, err = run(capsys, "build", str(tmp_path / "missing"), "--embedder", "hashing", "--db", str(tmp_path / "x.db"))
    assert code == 2
    assert "not found" in err
