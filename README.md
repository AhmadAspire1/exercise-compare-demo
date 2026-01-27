# Exercise Compare Demo (Local)

## What it does
- Upload tutorial + patient videos
- Extract pose using MediaPipe Pose
- Detect exercise (Squat / Shoulder Abduction / Heel Raises / Unknown)
- Estimate rep count
- Compare form metrics and compute similarity score (DTW)

## Setup
1) Create a virtual environment (recommended)
- Windows:
  python -m venv .venv
  .venv\Scripts\activate

- Mac/Linux:
  python3 -m venv .venv
  source .venv/bin/activate

2) Install dependencies
  pip install -r requirements.txt

3) Run
  streamlit run app.py

## Tips for good results
- Full body visible (head to feet)
- Stable camera and good lighting
- Similar camera angle for both videos
