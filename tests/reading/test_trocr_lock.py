"""A TrOCR read abandoned by a timed-out caller must not block every later read (audit H4)."""
import pytest

from s2c.reading import trocr


def test_a_held_read_lock_times_out(monkeypatch):
    monkeypatch.setattr(trocr, "READ_LOCK_S", 0.05)
    trocr._READ_LOCK.acquire()
    try:
        with pytest.raises(TimeoutError):
            trocr._run(None, None, "cpu", [object()], 4)
    finally:
        trocr._READ_LOCK.release()
