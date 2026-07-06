# Optional acquisition modules

The main reproducible package is the offline analysis pipeline. The files in
`acquisition/` are public, optional examples for generating auxiliary LSL
streams during a new recording.

These helpers were tested on Windows with Python 3.11. They have not been
validated on Linux or macOS.

## Included modules

- `audio_lsl_demo.py`: publishes a stereo audio waveform as `BCI_Audio` over
  LSL and can optionally play it locally.
- `eye_fixation_monitor/`: webcam-based coarse fixation monitor that publishes
  `EyeFix_Gaze` and `EyeFix_Markers`.
- `experiment_selector_demo.py`: small public GUI showing the type of module
  selector used before acquisition. It is a clean demo, not the full internal
  experiment runner.

## Installation

The analysis requirements remain in the repository root. Install acquisition
dependencies only when you need these optional modules:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r acquisition\requirements_acquisition.txt
```

## Audio stream example

```powershell
.\.venv\Scripts\python.exe acquisition\audio_lsl_demo.py --duration-s 5 --frequency-hz 1000 --side both --play
```

Expected LSL stream:

- `BCI_Audio`, type `Audio`, two float32 channels: `left`, `right`.

## Eye fixation example

```powershell
cd acquisition\eye_fixation_monitor
..\..\.venv\Scripts\python.exe main.py --mode preview
```

For a non-interactive smoke test without webcam:

```powershell
cd acquisition\eye_fixation_monitor
..\..\.venv\Scripts\python.exe main.py --mode headless --demo-mode simulated --duration-s 10
```

Expected LSL streams:

- `EyeFix_Gaze`, type `EyeGaze`.
- `EyeFix_Markers`, type `Markers`.

## Selector demo

```powershell
.\.venv\Scripts\python.exe acquisition\experiment_selector_demo.py --json-out outputs\selected_plan.json
```

The private acquisition runner used a richer selector tied to the laboratory
protocol. This public demo documents the intended workflow without exposing
machine-specific paths, participant information or unpublished raw session
logic.

## What not to upload

Do not commit raw XDF recordings, webcam videos, eye-tracker outputs,
participant-specific calibration files, LabRecorder files with names/dates, or
local paths from the acquisition computer.
