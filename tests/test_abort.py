import threading
import unittest

from forestcode.core.abort import Aborted, AbortSignal


class AbortSignalTest(unittest.TestCase):
    def test_starts_unset(self):
        signal = AbortSignal()
        self.assertFalse(signal.is_set())
        signal.raise_if_aborted()  # no raise

    def test_set_marks_is_set(self):
        signal = AbortSignal()
        signal.set()
        self.assertTrue(signal.is_set())

    def test_raise_if_aborted_raises_after_set(self):
        signal = AbortSignal()
        signal.set()
        with self.assertRaises(Aborted):
            signal.raise_if_aborted()

    def test_aborted_is_not_caught_by_except_exception(self):
        signal = AbortSignal()
        signal.set()
        caught_as_exception = False
        try:
            try:
                signal.raise_if_aborted()
            except Exception:  # noqa: BLE001
                caught_as_exception = True
        except Aborted:
            pass
        self.assertFalse(caught_as_exception)

    def test_on_abort_fires_on_set(self):
        signal = AbortSignal()
        calls: list[int] = []
        signal.on_abort(lambda: calls.append(1))
        signal.on_abort(lambda: calls.append(2))
        signal.set()
        self.assertEqual(calls, [1, 2])

    def test_on_abort_after_set_fires_immediately(self):
        signal = AbortSignal()
        signal.set()
        calls: list[int] = []
        signal.on_abort(lambda: calls.append(1))
        self.assertEqual(calls, [1])

    def test_set_is_idempotent(self):
        signal = AbortSignal()
        calls: list[int] = []
        signal.on_abort(lambda: calls.append(1))
        signal.set()
        signal.set()
        self.assertEqual(calls, [1])

    def test_callback_exception_is_isolated(self):
        signal = AbortSignal()
        calls: list[int] = []

        def boom() -> None:
            raise RuntimeError("boom")

        signal.on_abort(boom)
        signal.on_abort(lambda: calls.append(2))
        signal.set()  # must not propagate boom
        self.assertEqual(calls, [2])

    def test_concurrent_set_and_on_abort_callback_always_runs(self):
        # Stress the on_abort/set race: every registered callback must fire
        # exactly once whether set() wins or on_abort() wins.
        for _ in range(200):
            signal = AbortSignal()
            fired: list[int] = []
            lock = threading.Lock()

            def cb() -> None:
                with lock:
                    fired.append(1)

            start = threading.Event()

            def register() -> None:
                start.wait()
                signal.on_abort(cb)

            def trigger() -> None:
                start.wait()
                signal.set()

            t1 = threading.Thread(target=register)
            t2 = threading.Thread(target=trigger)
            t1.start()
            t2.start()
            start.set()
            t1.join()
            t2.join()
            self.assertEqual(len(fired), 1)


if __name__ == "__main__":
    unittest.main()
