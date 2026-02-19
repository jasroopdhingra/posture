# Posture Monitor

A real-time posture detection tool that uses your webcam and MediaPipe to detect when you are slouching and give live feedback — no API key required, runs fully offline.

## How it works

MediaPipe detects key landmarks on your face and shoulders every 2 seconds. The tool measures how high your nose and ears sit above your shoulder line (normalized by shoulder width) to determine whether you are sitting upright or slouching. The thresholds are calibrated for a standard laptop front-facing camera.

## Features

- **Real-time detection** — analyzes your posture every 2 seconds using your webcam
- **Skeleton overlay** — color-coded dots and lines on your nose, ears, and shoulders with a glow effect
- **Posture bar** — a GOOD → BAD bar showing your current slouch score at a glance
- **Duration timer** — shows how long you've held the current posture
- **Corner bracket border** — pulses red when slouching, solid green when upright
- **Advice strip** — coaching tip at the bottom of the frame

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run**:
   ```bash
   python posture_monitor.py
   ```

The pose model (`pose_landmarker_full.task`) is downloaded automatically on first run (~30 MB).

## Usage

- A window titled `Posture Monitor` will open, showing your webcam feed
- Sit with your **head and shoulders clearly visible** to the camera
- **Green** = upright, **Red** = slouching, **Yellow** = calibrating
- Press `q` to quit

## Configuration

You can tweak behavior via environment variables:

| Variable | Default | Description |
|---|---|---|
| `POSTURE_ANALYSIS_INTERVAL_SECS` | `2.0` | Seconds between posture checks |
| `POSTURE_MAX_ANALYSIS_WIDTH` | `640` | Max frame width sent to the model |
| `POSTURE_SMOOTHING_WINDOW` | `4` | Number of readings used for smoothing |

## Requirements

- Python 3.8+
- Webcam
- macOS / Linux / Windows
