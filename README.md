# Blink Rate & Facial Dimensions Analysis

This project was created for the **Mini Project assignment** for **CSC 8830**.

**Created by: Akshat Namdeo**

## Overview

This tool analyzes blink rate during a study session and estimates basic facial dimensions (face height, width, eye width, nose length, and mouth width) using MediaPipe Face Landmarker.

It processes `.mp4` videos and provides:
- Blink rate (blinks per second and per minute)
- Approximate facial measurements

## Project Structure

```
.
├── main_file_version.py       # Main script with interactive video selection
├── main.ipynb                 # Clean Jupyter Notebook version
├── videos/                    # Place your .mp4 videos here (not uploaded - large files)
├── results/                   # Analysis results are saved here (auto-created)
├── face_landmarker.task       # MediaPipe model (required)
└── README.md
```

## How to Run

### Option 1: Using the Python Script (Recommended)

1. Place your `.mp4` videos inside the `videos/` folder.
2. Make sure `face_landmarker.task` is in the project root.
3. Run the script:

```bash
python blink_face_analysis.py
```

4. When prompted, choose which videos to process:
   - Type `all` to process all videos, or
   - Enter numbers (e.g. `1,2`) to select specific videos.

Results will be displayed on screen and automatically saved as a timestamped `.txt` file in the `results/` folder.

### Option 2: Using the Jupyter Notebook

- Open `main.ipynb` in Jupyter Notebook / JupyterLab / VS Code.
- Run the cells in order.
- All outputs and plots will be shown directly in the notebook.

## Notes

- Two roughly 8-minute videos from my study trial were used for testing and demonstration.
- The `videos/` folder was **not uploaded** due to large file sizes.
- Place your own videos in the `videos/` folder before running the script or notebook.
- The facial dimension estimates are approximate (based on average adult face width scaling).

## Requirements

- Python 3.10+
- OpenCV (`cv2`)
- MediaPipe
- NumPy
- Matplotlib

Install dependencies with:
```bash
pip install opencv-python mediapipe numpy matplotlib
```

---

**Note:** Download the `face_landmarker.task` model from the official MediaPipe repository if it's missing.

