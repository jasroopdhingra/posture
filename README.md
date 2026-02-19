## Posture Monitor

This is a small Python tool that uses your webcam and the Gemini API to detect when you are slouching and give real‑time posture feedback.

### Features

- **Webcam posture detection**: Periodically sends frames from your camera to a Gemini model.
- **Slouching vs upright classification**: The model classifies your current posture and returns a confidence score.
- **Live overlay**: An OpenCV window displays your camera feed with posture status and short coaching tips.

### Setup

1. **Create and activate a virtual environment** (optional but recommended).
2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set your Gemini API key**:

   ```bash
   export GEMINI_API_KEY="your_api_key_here"
   ```

4. (Optional) Configure behavior via environment variables:

   - **`GEMINI_MODEL_NAME`**: Model to use (default: `gemini-2.5-flash`).
   - **`POSTURE_ANALYSIS_INTERVAL_SECS`**: Seconds between posture checks (default: `2.0`).
   - **`POSTURE_MAX_ANALYSIS_WIDTH`**: Max width (in pixels) used when resizing frames before sending them to the model (default: `640`).

### Usage

Run the posture monitor from the project directory:

```bash
python posture_monitor.py
```

- A window titled `Posture Monitor` will open showing your webcam feed.
- The overlay at the top will indicate:
  - **Posture**: `UPRIGHT`, `SLOUCHING`, or `UNKNOWN`.
  - **Confidence**: Model confidence as a percentage.
  - **Advice**: Short text tips on how to improve or maintain your posture.
- Press `q` to exit.

### Notes

- The script mirrors the webcam feed horizontally to feel more natural.
- Frames are resized before being sent to the model to reduce bandwidth and latency.
- If there is an API error, the last known posture state is kept and the advice field is updated with an error message.

