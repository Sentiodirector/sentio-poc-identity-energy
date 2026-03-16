"""
identity_energy.py
Sentio Mind · Project 1 · Named Face Identity + Energy Report

Copy this file to solution.py and fill in every TODO block.
Do not rename any function — the integration system calls them by name.
Run: python solution.py
"""

import cv2
import json
import base64
import time
import numpy as np
from pathlib import Path
from datetime import date

# ---------------------------------------------------------------------------
# CONFIG — edit these before running
# ---------------------------------------------------------------------------
KNOWN_FACES_DIR  = Path("known_faces")
VIDEO_PATH       = Path("video_sample_1.mov")
REPORT_HTML_OUT  = Path("report.html")
INTEGRATION_OUT  = Path("integration_output.json")

SCHOOL_NAME      = "Demo School"   # change this
MATCH_THRESHOLD  = 0.55            # 0.55 works well for CCTV quality; lower = stricter
MAX_KEYFRAMES    = 20              # how many frames to sample from the video


# ---------------------------------------------------------------------------
# STEP 1 — Load reference photos
# ---------------------------------------------------------------------------

def load_known_faces(folder: Path) -> dict:
    """
    Read every image from the folder.
    The filename without extension is the person's name.
    Encode each face with face_recognition.
    Return: { "Arjun Mehta": [128-d encoding, ...], ... }

    If a photo has no detectable face, print a warning and skip it.
    One person can have multiple photos — store all encodings in the list.

    TODO: implement using face_recognition.load_image_file + face_encodings
    """
    known = {}
    # TODO
    return known


# ---------------------------------------------------------------------------
# STEP 2 — Extract keyframes from video
# ---------------------------------------------------------------------------

def extract_keyframes(video_path: Path, max_frames: int) -> list:
    """
    Open the video, pull up to max_frames evenly spaced frames.
    Apply CLAHE on each frame to help with CCTV lighting.
    Return: [(frame_index, numpy_array), ...]

    TODO: use cv2.VideoCapture, compute step = total_frames // max_frames
    """
    frames = []
    # TODO
    return frames


# ---------------------------------------------------------------------------
# STEP 3 — Detect and match faces in one frame
# ---------------------------------------------------------------------------

def detect_and_match(frame: np.ndarray, known: dict, threshold: float) -> list:
    """
    Detect all faces in frame, compare each against known encodings.
    Return a list of dicts — one per detected face:
      {
        "name":       str,          real name OR "UNKNOWN_001" etc.
        "matched":    bool,
        "confidence": float,        1.0 = perfect match
        "bbox":       (x, y, w, h),
        "face_crop":  np.ndarray
      }

    TODO: use face_recognition.face_locations, face_encodings, compare_faces
    Assign UNKNOWN_001, UNKNOWN_002 etc. for faces that don't match anyone.
    """
    detections = []
    # TODO
    return detections


# ---------------------------------------------------------------------------
# STEP 4 — Energy components
# ---------------------------------------------------------------------------

def compute_face_brightness(face_crop: np.ndarray) -> float:
    """
    Grayscale mean pixel value scaled to 0–100.
    TODO: cv2.cvtColor BGR->GRAY, np.mean, divide by 2.55
    """
    # TODO
    return 50.0


def compute_eye_openness(face_crop: np.ndarray) -> float:
    """
    Average of (eye height / eye width) for left and right eye, scaled 0–100.
    Use MediaPipe Face Mesh to find eye landmarks.
    If MediaPipe fails, return 50.0 as a neutral fallback.
    TODO: implement
    """
    # TODO
    return 50.0


def compute_movement(prev_frame, curr_frame: np.ndarray, bbox: tuple) -> float:
    """
    Dense optical flow magnitude in the face bounding box region, scaled 0–100.
    If prev_frame is None (first frame) return 0.0.
    TODO: cv2.calcOpticalFlowFarneback on grayscale face crops
    """
    if prev_frame is None:
        return 0.0
    # TODO
    return 0.0


# ---------------------------------------------------------------------------
# STEP 5 — Aggregate across all frames into per-person summaries
# ---------------------------------------------------------------------------

def aggregate_persons(all_detections: list) -> list:
    """
    all_detections is a flat list of dicts from previous steps, each containing:
      name, matched, confidence, face_crop, brightness, eye_openness, movement, frame_idx

    Group by name.
    For each person:
      - average the three energy components
      - compute energy_score = brightness*0.35 + eye*0.30 + movement*0.35
      - pick the sharpest face_crop as the profile image
      - encode that crop as 240×240 base64 JPEG

    Return a list matching the "persons" array in identity_energy.json.
    TODO: implement
    """
    persons = []
    # TODO
    return persons


def encode_b64(img: np.ndarray, size=(240, 240)) -> str:
    """Resize to size, encode as JPEG, return base64 string."""
    resized = cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)
    _, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buf).decode("utf-8")


def verdict(score: float) -> str:
    """Returns 'high', 'moderate', or 'low'."""
    return "high" if score >= 75 else "moderate" if score >= 50 else "low"


# ---------------------------------------------------------------------------
# STEP 6 — HTML report
# ---------------------------------------------------------------------------

def generate_report(persons: list, output_path: Path):
    """
    Write a self-contained HTML file to output_path.
    For each person include: profile photo (embedded), name, energy bar, breakdown, verdict.
    Inline CSS only — no CDN, no external links, must work offline.
    Keep it clean — school staff will read this, not engineers.
    TODO: build HTML string, write to file
    """
    # TODO
    pass


# ---------------------------------------------------------------------------
# STEP 7 — Integration JSON
# ---------------------------------------------------------------------------

def write_integration_json(persons: list, output_path: Path,
                            video_name: str, processing_time: float):
    """
    Write integration_output.json following identity_energy.json exactly.
    Do not add or remove top-level keys.
    TODO: implement
    """
    output = {
        "source": "p1_identity_energy",
        "school": SCHOOL_NAME,
        "date": str(date.today()),
        "video_file": video_name,
        "total_persons_matched": sum(1 for p in persons if p.get("matched")),
        "total_persons_unknown": sum(1 for p in persons if not p.get("matched")),
        "processing_time_sec": processing_time,
        "persons": persons,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()

    print("Step 1 — loading known faces ...")
    known = load_known_faces(KNOWN_FACES_DIR)
    if not known:
        print("  ERROR: no faces loaded. Check known_faces/ has images named like 'Arjun Mehta.jpg'")
        raise SystemExit(1)
    print(f"  {len(known)} persons: {', '.join(list(known.keys())[:6])}")

    print("Step 2 — extracting keyframes ...")
    frames = extract_keyframes(VIDEO_PATH, MAX_KEYFRAMES)
    print(f"  {len(frames)} frames extracted")

    print("Step 3 & 4 — detecting + scoring faces ...")
    all_detections = []
    prev_frame = None
    for frame_idx, frame in frames:
        detections = detect_and_match(frame, known, MATCH_THRESHOLD)
        for d in detections:
            d["frame_idx"]    = frame_idx
            d["brightness"]   = compute_face_brightness(d["face_crop"])
            d["eye_openness"] = compute_eye_openness(d["face_crop"])
            d["movement"]     = compute_movement(prev_frame, frame, d["bbox"])
        all_detections.extend(detections)
        prev_frame = frame

    print("Step 5 — aggregating per-person ...")
    persons = aggregate_persons(all_detections)

    t1 = round(time.time() - t0, 2)

    print("Step 6 — writing report.html ...")
    generate_report(persons, REPORT_HTML_OUT)

    print("Step 7 — writing integration_output.json ...")
    write_integration_json(persons, INTEGRATION_OUT, str(VIDEO_PATH), t1)

    print()
    print("=" * 50)
    print(f"  Finished in {t1}s")
    print(f"  Persons found: {len(persons)}")
    for p in persons:
        print(f"    {p['name']:30s}  energy {p['energy_score']:5.1f}  ({p['verdict']})")
    print(f"  report.html              -> {REPORT_HTML_OUT}")
    print(f"  integration_output.json  -> {INTEGRATION_OUT}")
    print("=" * 50)
