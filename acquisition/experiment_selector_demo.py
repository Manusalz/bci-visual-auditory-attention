"""Small public demo of the experiment/module selector used before acquisition."""

from __future__ import annotations

import argparse
import json
import tkinter as tk
from dataclasses import asdict, dataclass
from tkinter import ttk


@dataclass
class SelectedPlan:
    baseline: bool
    visual: bool
    audio2: bool
    audio4: bool
    assr: bool
    eye_tracker: bool
    notes: str


def run_gui() -> SelectedPlan:
    root = tk.Tk()
    root.title("BCI module selector demo")
    root.resizable(False, False)

    variables = {
        "baseline": tk.BooleanVar(value=True),
        "visual": tk.BooleanVar(value=True),
        "audio2": tk.BooleanVar(value=True),
        "audio4": tk.BooleanVar(value=True),
        "assr": tk.BooleanVar(value=False),
        "eye_tracker": tk.BooleanVar(value=False),
    }
    notes = tk.StringVar(value="")
    result: dict[str, SelectedPlan | None] = {"plan": None}

    frame = ttk.Frame(root, padding=16)
    frame.grid(row=0, column=0, sticky="nsew")
    ttk.Label(frame, text="Select modules for this run").grid(row=0, column=0, columnspan=2, sticky="w")

    labels = {
        "baseline": "Baseline",
        "visual": "Visual attention",
        "audio2": "Auditory 2-stream",
        "audio4": "Auditory 4-stream",
        "assr": "ASSR check",
        "eye_tracker": "Eye fixation monitor",
    }
    for row, key in enumerate(labels, start=1):
        ttk.Checkbutton(frame, text=labels[key], variable=variables[key]).grid(row=row, column=0, sticky="w")

    ttk.Label(frame, text="Notes").grid(row=7, column=0, sticky="w", pady=(8, 0))
    ttk.Entry(frame, textvariable=notes, width=40).grid(row=8, column=0, columnspan=2, sticky="ew")

    def accept() -> None:
        result["plan"] = SelectedPlan(
            baseline=variables["baseline"].get(),
            visual=variables["visual"].get(),
            audio2=variables["audio2"].get(),
            audio4=variables["audio4"].get(),
            assr=variables["assr"].get(),
            eye_tracker=variables["eye_tracker"].get(),
            notes=notes.get(),
        )
        root.destroy()

    def cancel() -> None:
        result["plan"] = SelectedPlan(False, False, False, False, False, False, "cancelled")
        root.destroy()

    buttons = ttk.Frame(frame)
    buttons.grid(row=9, column=0, columnspan=2, sticky="e", pady=(12, 0))
    ttk.Button(buttons, text="Cancel", command=cancel).grid(row=0, column=0, padx=(0, 8))
    ttk.Button(buttons, text="Start", command=accept).grid(row=0, column=1)

    root.mainloop()
    return result["plan"] or SelectedPlan(False, False, False, False, False, False, "closed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Public GUI demo for choosing acquisition modules.")
    parser.add_argument("--json-out", default="", help="Optional path to save the selected plan as JSON.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    plan = run_gui()
    payload = asdict(plan)
    print(json.dumps(payload, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
