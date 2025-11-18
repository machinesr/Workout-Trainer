# AI Personal Trainer

> **Version 2 Released**
>
> A more user-friendly version with full frontend integration is now available:
> **PermaFit v2** — [https://github.com/5calvinw/PermaFit](https://github.com/5calvinw/PermaFit)
>
> This repository represents **Version 1**, built fully in VSCode using OpenCV and MediaPipe, focusing on CV logic and exercise analysis.

## Overview

The **AI Personal Trainer** is a computer vision application designed to assist users in performing physical exercises with correct form and timing. Using MediaPipe Pose estimation and OpenCV, it analyzes real-time video to track body landmarks, calculate joint angles, and provide immediate visual feedback.

Unlike simple repetition counters, this tool monitors movement tempo (concentric, hold, eccentric phases) and enforces strict form constraints to ensure efficient and safe training.

---

## Key Features

* **Real-Time Pose Estimation** — Detects 33 3D body landmarks with high precision via MediaPipe.
* **Form Correction** — Analyzes joint positions and angles to detect form errors (e.g., "Squat too deep", "Back not straight").
* **Tempo Monitoring** — Ensures controlled movement speed for different lifting phases.
* **Repetition Counting** — Separates "Good" reps from "Bad" reps based on form and timing.
* **Multi-Exercise Support** — Includes logic for Bicep Curls, Squats, Wall Push-ups, Glute Bridges, and Seated Leg Raises.
* **Visual Feedback Interface** — Displays feedback, rep counters, and progress bars directly on screen.

---

## Technical Requirements

**Built with Python**, using:

* Python 3.12
* OpenCV (opencv-python)
* MediaPipe
* NumPy

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/machinesr/Workout-Trainer.git
cd ai-personal-trainer
```

### Create a Virtual Environment (Recommended)

```bash
python -m venv ai-trainer-env
# Windows
ai-trainer-env\Scripts\activate
# macOS/Linux
source ai-trainer-env/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Connect a Webcam

The default camera index is `1`. For built-in laptop cameras, edit `AiTrainer.py`:

```python
cap = cv2.VideoCapture(0)
```

### Run the Application

```bash
python AiTrainer.py
```

### Positioning

Ensure your full body (or target body region) is visible. The system will prompt adjustments when landmarks are unclear.

---

## Controls

| Key | Action                     |
| --- | -------------------------- |
| 1   | Switch to Bicep Curl       |
| 2   | Switch to Squat            |
| 3   | Switch to Wall Push-up     |
| 4   | Switch to Glute Bridge     |
| 5   | Switch to Seated Leg Raise |
| r   | Reset exercise statistics  |
| q   | Quit application           |

---

## Supported Exercises & Metrics

### 1. Bicep Curl

* Primary: Elbow flexion
* Form: Detects shoulder drift
* Visibility: Right upper body

### 2. Squat

* Primary: Knee angle & hip depth
* Form: Chest angle & squat depth
* Visibility: Full body

### 3. Wall Push-up

* Primary: Elbow flexion
* Form: Straight back, proper elbow tuck
* Visibility: Upper body + hips

### 4. Glute Bridge

* Primary: Hip extension
* Form: Overextension & feet placement
* Visibility: Full side profile

### 5. Seated Leg Raise

* Primary: Hip flexion
* Form: Upright posture, stable support leg
* Visibility: Full side profile

---

## Project Structure

```
AiTrainer.py        # Main entry, state machine, tempo stages, UI overlay
PoseModule.py       # MediaPipe pose utilities & angle calculation
requirements.txt    # Dependency list
```

---

## Configuration

Modify `EXERCISE_CONFIG` in `AiTrainer.py` to adjust:

* Timing phase durations (hold, concentric, eccentric)
* Angle thresholds
* Form correctness constraints

---

## License

Open-source for personal and educational use.
