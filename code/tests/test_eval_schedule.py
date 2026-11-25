"""build_schedule with warm-up days (evaluation of a checkpoint, train_full.py --init-from)."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("train_full", ROOT / "train_full.py")
train_full = importlib.util.module_from_spec(spec)
sys.modules["train_full"] = train_full
spec.loader.exec_module(train_full)
build_schedule = train_full.build_schedule


def test_schedule_without_warm_days_is_unchanged():
    sched = build_schedule([6, 7], [13], None)
    assert len(sched) == 3 * 24
    assert all(len(s) == 6 for s in sched)
    assert [s[4] for s in sched] == [True] * 48 + [False] * 24
    assert [s[5] for s in sched] == [False] * 48 + [True] * 24
    assert sched[0][:4] == (2026, 3, 6, 0) and sched[-1][:4] == (2026, 3, 13, 23)


def test_warm_days_run_between_training_and_validation_without_recording():
    sched = build_schedule([], [24, 25, 26], None, warm_days=[22, 23])
    assert len(sched) == 5 * 24
    assert not any(s[4] for s in sched)                       # nothing is learned
    assert [s[5] for s in sched] == [False] * 48 + [True] * 72  # only the validation hours are recorded
    days = [s[2] for s in sched]
    assert days == sorted(days) and days[0] == 22 and days[-1] == 26


def test_max_hours_truncates_every_phase():
    sched = build_schedule([6], [13], 2, warm_days=[12])
    assert [(s[2], s[3], s[4], s[5]) for s in sched] == [
        (6, 0, True, False), (6, 1, True, False),
        (12, 0, False, False), (12, 1, False, False),
        (13, 0, False, True), (13, 1, False, True),
    ]
