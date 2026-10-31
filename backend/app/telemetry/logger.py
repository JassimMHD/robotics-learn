"""Append-only JSONL event log (one file per day)."""

import json
import threading
from datetime import date
from pathlib import Path


class EventLog:
    def __init__(self, directory: Path | None):
        self.directory = directory
        self._lock = threading.Lock()
        if directory:
            directory.mkdir(parents=True, exist_ok=True)

    def write(self, event: dict) -> None:
        if not self.directory:
            return
        path = self.directory / f"events-{date.today().isoformat()}.jsonl"
        line = json.dumps(event, separators=(",", ":"), default=str)
        with self._lock, path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
