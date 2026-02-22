# Posture Monitor

Real-time posture detection using your webcam. Detects slouching and gives live feedback — no API key, runs fully offline.

## Setup (one time)

You need Python 3 installed. [Download it here](https://www.python.org/downloads/) if you don't have it.

Open Terminal and run:

```bash
# 1. Download the project
git clone https://github.com/jasroopdhingra/posture.git
cd posture

# 2. Install dependencies
pip3 install -r requirements.txt
```

> The pose model (~30 MB) downloads automatically the first time you run the app.

## Run it

```bash
python3 posture_monitor.py
```

A window will open showing your webcam feed. Press `q` to quit.

## Tips

- Sit so your **head and shoulders are fully visible** to the camera
- **Green** = good posture · **Red** = slouching · **Yellow** = calibrating
- Takes ~4 seconds to calibrate on first launch

## Features

- Skeleton overlay with color-coded dots on your nose, ears, and shoulders
- Posture bar showing your current slouch score (GOOD → BAD)
- Duration timer showing how long you've held the current posture
- Pulsing red border when slouching, solid green when upright
- Coaching tip at the bottom of the frame
