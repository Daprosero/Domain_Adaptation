import json

from MIL_CREDA_EEG.progress import Progress, probe


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_step_prints_line_writes_row_and_estimates_from_measured_rate(tmp_path, capsys):
    c = Clock()
    p = Progress("u1", 4, tmp_path, clock=c)
    c.t = 120.0
    row = p.step("search 1")
    assert row["leftMin"] == 6.0 and row["elapsedMin"] == 2.0
    assert "[u1] search 1 1/4 · 2.0 min · ~6 min left" in capsys.readouterr().out
    rows = [json.loads(l) for l in (tmp_path / "progress.jsonl").read_text().splitlines()]
    assert rows == [row]


def test_weights_and_added_work_change_the_estimate(tmp_path):
    c = Clock()
    p = Progress("u", 2, tmp_path, clock=c)
    c.t = 60.0
    assert p.step("a", weight=2)["leftMin"] == 0.0
    p.add(2)
    c.t = 120.0
    assert p.step("b")["leftMin"] == 0.67


def test_probe_ticks_the_declared_number_of_times(tmp_path, capsys):
    out = probe(minutes=1, every=30, workdir=tmp_path, sleep=lambda s: None)
    assert out["ran"] == ["log-probe"]
    assert capsys.readouterr().out.count("[log-probe] tick") == 2
