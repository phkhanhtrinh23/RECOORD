import json

from recoord.cli import main


def test_datasets_lists_the_specs(capsys):
    assert main(["datasets"]) == 0
    assert "catalog" in capsys.readouterr().out


def test_run_writes_json_and_csv(tmp_path, capsys):
    assert main(["run", "demo", "--out", str(tmp_path), "--methods", "recoord", "lww"]) == 0
    payload = json.loads((tmp_path / "demo.json").read_text())
    assert {r["method"] for r in payload["rows"]} == {"recoord", "lww"}
    assert (tmp_path / "demo.csv").exists()


def test_sweep_writes_its_own_file(tmp_path):
    assert main(["sweep", "ablation", "demo", "--out", str(tmp_path), "--quiet"]) == 0
    assert (tmp_path / "sweep-ablation-demo.csv").exists()


def test_report_aggregates_what_run_wrote(tmp_path, capsys):
    main(["run", "demo", "--out", str(tmp_path), "--quiet"])
    capsys.readouterr()
    assert main(["report", str(tmp_path), "--compare", "lww"]) == 0
    out = capsys.readouterr().out
    assert "recoord" in out and "per_dataset" in out


def test_check_passes(capsys):
    assert main(["check", "--limit", "30"]) == 0
    assert "FAIL" not in capsys.readouterr().out


def test_unknown_method_is_rejected(tmp_path, capsys):
    assert main(["run", "demo", "--out", str(tmp_path), "--methods", "nope"]) == 1
    assert "nope" in capsys.readouterr().err


def test_unknown_dataset_is_rejected(tmp_path, capsys):
    assert main(["run", "nosuch", "--out", str(tmp_path)]) == 1
    assert "no dataset spec" in capsys.readouterr().err
