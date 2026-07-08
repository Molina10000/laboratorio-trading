from __future__ import annotations

import unittest

from watcher import _should_alert


class WatcherAlertTests(unittest.TestCase):
    def test_alerts_when_entry_becomes_enabled(self) -> None:
        previous = {"entrada": "NO HABILITADA", "semaforo": "VIGILAR"}
        current = {"entrada": "HABILITADA", "semaforo": "VIGILAR"}
        self.assertTrue(_should_alert(previous, current))

    def test_alerts_when_semaforo_becomes_operativo(self) -> None:
        previous = {"entrada": "NO HABILITADA", "semaforo": "VIGILAR"}
        current = {"entrada": "NO HABILITADA", "semaforo": "OPERATIVO"}
        self.assertTrue(_should_alert(previous, current))

    def test_does_not_alert_when_signal_is_already_active(self) -> None:
        previous = {"entrada": "HABILITADA", "semaforo": "OPERATIVO"}
        current = {"entrada": "HABILITADA", "semaforo": "OPERATIVO"}
        self.assertFalse(_should_alert(previous, current))


if __name__ == "__main__":
    unittest.main()
