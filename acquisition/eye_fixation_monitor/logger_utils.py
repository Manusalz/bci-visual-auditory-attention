from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any


def create_output_dir(output_root: Path, mode: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    suffix = 0
    while True:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        base_name = f"{timestamp}_{mode}" if suffix == 0 else f"{timestamp}_{mode}_{suffix:02d}"
        out_dir = output_root / base_name
        try:
            out_dir.mkdir(parents=True, exist_ok=False)
            return out_dir
        except FileExistsError:
            suffix += 1


def setup_session_logger(log_path: Path, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("eye_fix_monitor")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


class SafeCsvWriter:
    """CSV sink that flushes aggressively to reduce data loss on crashes."""

    def __init__(self, csv_path: Path, fieldnames: list[str]) -> None:
        self.csv_path = csv_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.csv_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()
        self._file.flush()

    def write_row(self, row: dict[str, Any]) -> None:
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        try:
            self._file.flush()
        finally:
            self._file.close()
