"""
solution.py
Sentio Mind · Project 1 · Named Face Identity + Energy Report
"""

import cv2
import json
import base64
import time
import numpy as np
from pathlib import Path
from datetime import date
import face_recognition
import mediapipe as mp
from collections import defaultdict

# ---------------------------------------------------------------------------
# CONFIG — edit these before running
# ---------------------------------------------------------------------------
KNOWN_FACES_DIR = Path("known_faces")
VIDEO_PATH      = Path("video_sample_1.mov")
REPORT_HTML_OUT = Path("report.html")
INTEGRATION_OUT = Path("integration_output.json")
SCHOOL_NAME     = "Demo School"   # change this
MATCH_THRESHOLD = 0.55            # 0.55 works well for CCTV quality; lower = stricter
MAX_KEYFRAMES   = 20              # how many frames to sample from the video


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
    """
    known = {}
    if not folder.exists():
        print(f"  ERROR: folder not found: {folder}")
        return known

    for img_path in folder.glob("*"):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
            continue
        name = img_path.stem  # full filename stem = person's name
        try:
            img = face_recognition.load_image_file(str(img_path))
            encodings = face_recognition.face_encodings(img)
            if not encodings:
                print(f"  WARNING: no face detected in {img_path.name}, skipping")
                continue
            if name not in known:
                known[name] = []
            known[name].extend(encodings)
            print(f"  [Loaded] {name}")
        except Exception as e:
            print(f"  WARNING: error loading {img_path.name}: {e}, skipping")

    return known


# ---------------------------------------------------------------------------
# STEP 2 — Extract keyframes from video
# ---------------------------------------------------------------------------

def extract_keyframes(video_path: Path, max_frames: int) -> list:
    """
    Open the video, pull up to max_frames evenly spaced frames.
    Apply CLAHE in LAB colour space to help with CCTV lighting.
    Return: [(frame_index, numpy_array), ...]
    """
    frames = []
    if not video_path.exists():
        print(f"  ERROR: video not found: {video_path}")
        return frames

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0 or max_frames <= 0:
        cap.release()
        return frames

    step = max(1, total_frames // max_frames)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break

        # LAB-space CLAHE — preserves colour, only boosts luminance channel
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        enhanced = cv2.merge((l, a, b))
        frame = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        frames.append((i, frame))
        if len(frames) >= max_frames:
            break

    cap.release()
    return frames


# ---------------------------------------------------------------------------
# STEP 3 — Detect and match faces in one frame
# ---------------------------------------------------------------------------

def detect_and_match(frame: np.ndarray, known: dict, threshold: float) -> list:
    """
    Detect all faces in frame using HOG model, compare each against known encodings.
    Returns a list of dicts — one per detected face:
      {
        "name":       str,          real name OR "UNKNOWN_001" etc.
        "matched":    bool,
        "confidence": float,        1.0 = perfect match
        "bbox":       (x, y, w, h),
        "face_crop":  np.ndarray
      }
    """
    detections = []

    # Downscale for faster HOG detection, scale coords back after
    small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    face_locs = face_recognition.face_locations(rgb, model="hog")
    face_encs = face_recognition.face_encodings(rgb, face_locs)

    # Scale bounding boxes back to original resolution
    face_locs = [(t*2, r*2, b*2, l*2) for (t, r, b, l) in face_locs]

    unknown_counter = 0

    for (top, right, bottom, left), encoding in zip(face_locs, face_encs):
        face_h = bottom - top
        face_w = right  - left

        # DROP the old 40px filter — now handle faces as small as 20px
        if face_h < 20 or face_w < 20:
            continue

        # For small faces (20–60px), upscale crop before encoding
        # so face_recognition has enough pixels to work with
        h, w = frame.shape[:2]
        x  = max(0, left);   y  = max(0, top)
        x2 = min(w, right);  y2 = min(h, bottom)
        face_crop = frame[y:y2, x:x2].copy()

        if face_h < 60 or face_w < 60:
            scale     = 120 / max(face_h, face_w)
            face_crop_enc = cv2.resize(face_crop, (0, 0), fx=scale, fy=scale,
                                        interpolation=cv2.INTER_CUBIC)
            rgb_crop  = cv2.cvtColor(face_crop_enc, cv2.COLOR_BGR2RGB)
            small_enc = face_recognition.face_encodings(rgb_crop)
            if small_enc:
                encoding = small_enc[0]   # re-encode from upscaled crop


        name            = None
        matched         = False
        best_distance   = 1.0

        for person_name, person_encs in known.items():
            distances = face_recognition.face_distance(person_encs, encoding)
            min_dist  = float(np.min(distances))
            if min_dist < best_distance:
                best_distance = min_dist
                name          = person_name

        if name is not None and best_distance <= threshold:
            matched = True
        else:
            unknown_counter += 1
            name    = f"UNKNOWN_{unknown_counter:03d}"
            matched = False

        confidence = float(1.0 - best_distance)

        detections.append({
            "name":      name,
            "matched":   matched,
            "confidence": round(confidence, 4),
            "bbox":      (x, y, x2 - x, y2 - y),
            "face_crop": face_crop,
        })

    return detections


# ---------------------------------------------------------------------------
# STEP 4 — Energy components
# ---------------------------------------------------------------------------

def compute_face_brightness(face_crop: np.ndarray) -> float:
    """Grayscale mean pixel value scaled to 0–100."""
    if face_crop is None or face_crop.size == 0:
        return 50.0
    gray     = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    mean_val = float(np.mean(gray))
    return float(np.clip(mean_val / 2.55, 0.0, 100.0))


def compute_eye_openness(face_crop: np.ndarray) -> float:
    """
    Eye Aspect Ratio (EAR) via MediaPipe Face Mesh, scaled 0–100.
    Falls back to 50.0 if detection fails.
    """
    if face_crop is None or face_crop.size == 0:
        return 50.0
    try:
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                    refine_landmarks=False) as face_mesh:
            rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            results = face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                return 50.0

            lm = results.multi_face_landmarks[0]

            def ear(indices):
                pts = [(lm.landmark[i].x * w, lm.landmark[i].y * h)
                       for i in indices]
                # vertical distances
                v1     = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
                v2     = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
                # horizontal distance
                h_dist = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
                if h_dist < 1e-6:
                    return 0.0
                return (v1 + v2) / (2.0 * h_dist)

            # Left eye landmark indices (MediaPipe 468-point mesh)
            left_ear  = ear([33, 160, 158, 133, 153, 144])
            # Right eye landmark indices
            right_ear = ear([362, 385, 387, 263, 373, 380])

            avg_ear = (left_ear + right_ear) / 2.0
            # EAR open ~0.25–0.30, closed ~0.05–0.10 → scale ×250 → 0–100
            score = float(np.clip(avg_ear * 250.0, 0.0, 100.0))
            return score
    except Exception:
        return 50.0


def compute_movement(prev_frame, curr_frame: np.ndarray, bbox: tuple) -> float:
    """
    Dense optical flow (Farneback) magnitude in the face bounding box, scaled 0–100.
    Returns 0.0 if prev_frame is None (first frame).
    """
    if prev_frame is None:
        return 0.0

    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return 0.0

    prev_crop = prev_frame[y:y+h, x:x+w]
    curr_crop = curr_frame[y:y+h, x:x+w]

    if prev_crop.shape != curr_crop.shape or prev_crop.size == 0:
        return 0.0

    prev_g = cv2.cvtColor(prev_crop, cv2.COLOR_BGR2GRAY)
    curr_g = cv2.cvtColor(curr_crop, cv2.COLOR_BGR2GRAY)

    flow      = cv2.calcOpticalFlowFarneback(prev_g, curr_g, None,
                                              0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _    = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mean_mag  = float(np.mean(mag))
    return float(min(100.0, mean_mag * 15.0))


# ---------------------------------------------------------------------------
# STEP 5 — Aggregate across all frames into per-person summaries
# ---------------------------------------------------------------------------

def aggregate_persons(all_detections: list) -> list:
    """
    Group detections by name, average energy components, pick sharpest crop,
    encode as base64 JPEG, return list matching the integration JSON schema.
    """
    by_name = defaultdict(list)
    for d in all_detections:
        by_name[d["name"]].append(d)

    persons = []

    for name, group in by_name.items():
        n              = len(group)
        brightness_avg = sum(d["brightness"]   for d in group) / n
        eye_avg        = sum(d["eye_openness"]  for d in group) / n
        movement_avg   = sum(d["movement"]      for d in group) / n

        energy_score   = (brightness_avg * 0.35
                          + eye_avg       * 0.30
                          + movement_avg  * 0.35)

        matched        = any(d["matched"] for d in group)
        confidence_avg = sum(d["confidence"] for d in group) / n

        # Pick the sharpest face crop using Laplacian variance
        def sharpness(crop):
            if crop is None or crop.size == 0:
                return 0.0
            g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            return float(cv2.Laplacian(g, cv2.CV_64F).var())

        best_crop    = max(group, key=lambda d: sharpness(d["face_crop"]))
        profile_b64  = encode_b64(best_crop["face_crop"])

        persons.append({
            "name":          name,
            "matched":       matched,
            "confidence":    round(confidence_avg, 4),
            "energy_score":  round(energy_score,   2),
            "verdict":       verdict(energy_score),
            "profile_image": profile_b64,
            "brightness":    round(brightness_avg, 2),
            "eye_openness":  round(eye_avg,         2),
            "movement":      round(movement_avg,    2),
        })

    # Sort: matched persons first, then unknowns, each group by energy desc
    persons.sort(key=lambda p: (not p["matched"], -p["energy_score"]))
    return persons


def encode_b64(img: np.ndarray, size=(240, 240)) -> str:
    """Resize to size, encode as JPEG, return base64 string."""
    if img is None or img.size == 0:
        blank = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", blank)
        return base64.b64encode(buf).decode("utf-8")
    resized = cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)
    _, buf  = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buf).decode("utf-8")


def verdict(score: float) -> str:
    """Returns 'high', 'moderate', or 'low'."""
    return "high" if score >= 75 else "moderate" if score >= 50 else "low"


# ---------------------------------------------------------------------------
# STEP 6 — HTML report
# ---------------------------------------------------------------------------

def generate_report(persons: list, output_path: Path):
    """
    Write a self-contained HTML file — works fully offline.
    Includes: profile photo, name, energy bar, score breakdown, verdict badge.
    """
    matched_count = sum(1 for p in persons if p["matched"])
    unknown_count = len(persons) - matched_count

    verdict_color = {"high": "#0a8a4a", "moderate": "#b06000", "low": "#c0392b"}
    verdict_bg    = {"high": "#e8f7ef", "moderate": "#fff3e0", "low": "#fdecea"}

    rows = ""
    for p in persons:
        fill   = min(100, max(0, p["energy_score"]))
        vcolor = verdict_color.get(p["verdict"], "#555")
        vbg    = verdict_bg.get(p["verdict"],    "#eee")
        bar_c  = vcolor

        rows += f"""
        <div class="card">
          <img src="data:image/jpeg;base64,{p['profile_image']}" alt="{p['name']}" />
          <div class="info">
            <div class="name-row">
              <span class="pname">{p['name']}</span>
              <span class="verdict-badge" style="color:{vcolor};background:{vbg};">
                {p['verdict'].upper()} ENERGY
              </span>
            </div>
            <div class="score-label">Energy Score: <strong>{p['energy_score']:.1f} / 100</strong></div>
            <div class="bar-wrap">
              <div class="bar-fill" style="width:{fill}%;background:{bar_c};"></div>
            </div>
            <div class="breakdown">
              <span>☀ Brightness&nbsp;<strong>{p['brightness']:.1f}</strong></span>
              <span>👁 Eye Openness&nbsp;<strong>{p['eye_openness']:.1f}</strong></span>
              <span>⚡ Movement&nbsp;<strong>{p['movement']:.1f}</strong></span>
              <span>🎯 Confidence&nbsp;<strong>{p['confidence']*100:.0f}%</strong></span>
            </div>
          </div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{SCHOOL_NAME} — Identity &amp; Energy Report</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: system-ui, -apple-system, sans-serif;
    background: #f5f6fa;
    color: #2d2d2d;
    padding: 2rem 1rem;
  }}
  .page {{ max-width: 860px; margin: 0 auto; }}
  header {{ margin-bottom: 2rem; }}
  header h1 {{ font-size: 1.6rem; font-weight: 600; color: #1a1a2e; }}
  header p  {{ font-size: 0.88rem; color: #666; margin-top: 0.3rem; }}
  .stats {{
    display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 1.8rem;
  }}
  .stat {{
    background: #fff; border-radius: 10px; padding: 14px 20px;
    border: 1px solid #e2e5ee; flex: 1; min-width: 130px;
  }}
  .stat .num {{ font-size: 1.8rem; font-weight: 600; color: #1a1a2e; }}
  .stat .lbl {{ font-size: 0.78rem; color: #888; margin-top: 2px; }}
  .card {{
    background: #fff; border-radius: 12px; border: 1px solid #e2e5ee;
    padding: 1.2rem; display: flex; gap: 1.2rem; margin-bottom: 1rem;
    align-items: flex-start;
  }}
  .card img {{
    width: 100px; height: 100px; object-fit: cover;
    border-radius: 8px; flex-shrink: 0;
    border: 2px solid #e2e5ee;
  }}
  .info {{ flex: 1; }}
  .name-row {{
    display: flex; align-items: center; gap: 10px;
    flex-wrap: wrap; margin-bottom: 6px;
  }}
  .pname {{ font-size: 1.05rem; font-weight: 600; color: #1a1a2e; }}
  .verdict-badge {{
    font-size: 0.72rem; font-weight: 700; padding: 3px 10px;
    border-radius: 20px; letter-spacing: 0.04em;
  }}
  .score-label {{ font-size: 0.82rem; color: #555; margin-bottom: 5px; }}
  .bar-wrap {{
    height: 10px; background: #eef0f5; border-radius: 6px;
    overflow: hidden; margin-bottom: 8px;
  }}
  .bar-fill {{ height: 100%; border-radius: 6px; transition: width 0.3s; }}
  .breakdown {{
    display: flex; gap: 14px; flex-wrap: wrap; font-size: 0.8rem; color: #555;
  }}
  .breakdown span {{ white-space: nowrap; }}
  .breakdown strong {{ color: #1a1a2e; }}
  footer {{
    text-align: center; font-size: 0.78rem; color: #aaa;
    margin-top: 2.5rem; padding-top: 1rem;
    border-top: 1px solid #e2e5ee;
  }}
</style>
</head>
<body>
<div class="page">
  <header>
    <h1>{SCHOOL_NAME} — Identity &amp; Energy Report</h1>
    <p>Generated: {date.today()} &nbsp;·&nbsp; Video: {VIDEO_PATH.name}
       &nbsp;·&nbsp; Total persons: {len(persons)}</p>
  </header>
  <div class="stats">
    <div class="stat">
      <div class="num">{len(persons)}</div>
      <div class="lbl">Total persons</div>
    </div>
    <div class="stat">
      <div class="num" style="color:#0a8a4a">{matched_count}</div>
      <div class="lbl">Matched (named)</div>
    </div>
    <div class="stat">
      <div class="num" style="color:#c0392b">{unknown_count}</div>
      <div class="lbl">Unknown</div>
    </div>
    <div class="stat">
      <div class="num" style="color:#b06000">
        {sum(1 for p in persons if p['verdict']=='high')}
      </div>
      <div class="lbl">High energy</div>
    </div>
  </div>
  {rows}
  <footer>Sentio Mind · {SCHOOL_NAME} · {date.today()}</footer>
</div>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  report written → {output_path}")


# ---------------------------------------------------------------------------
# STEP 7 — Integration JSON
# ---------------------------------------------------------------------------

def write_integration_json(persons: list, output_path: Path,
                            video_name: str, processing_time: float):
    """
    Write integration_output.json following identity_energy.json exactly.
    Does not add or remove any top-level keys.
    """
    output = {
        "source":                  "p1_identity_energy",
        "school":                  SCHOOL_NAME,
        "date":                    str(date.today()),
        "video_file":              video_name,
        "total_persons_matched":   sum(1 for p in persons if p.get("matched")),
        "total_persons_unknown":   sum(1 for p in persons if not p.get("matched")),
        "processing_time_sec":     processing_time,
        "persons":                 persons,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"  JSON written   → {output_path}")


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
    if not frames:
        print("  ERROR: no frames extracted. Check VIDEO_PATH.")
        raise SystemExit(1)
    print(f"  {len(frames)} frames extracted")

    print("Step 3 & 4 — detecting + scoring faces ...")
    all_detections = []
    prev_frame     = None
    for frame_idx, frame in frames:
        detections = detect_and_match(frame, known, MATCH_THRESHOLD)
        for d in detections:
            d["frame_idx"]    = frame_idx
            d["brightness"]   = compute_face_brightness(d["face_crop"])
            d["eye_openness"] = compute_eye_openness(d["face_crop"])
            d["movement"]     = compute_movement(prev_frame, frame, d["bbox"])
        all_detections.extend(detections)
        prev_frame = frame
    print(f"  {len(all_detections)} total face detections")

    print("Step 5 — aggregating per-person ...")
    persons = aggregate_persons(all_detections)

    t1 = round(time.time() - t0, 2)

    print("Step 6 — writing report.html ...")
    generate_report(persons, REPORT_HTML_OUT)

    print("Step 7 — writing integration_output.json ...")
    write_integration_json(persons, INTEGRATION_OUT, str(VIDEO_PATH), t1)

    print()
    print("=" * 55)
    print(f"  Finished in {t1}s")
    print(f"  Persons found : {len(persons)}")
    for p in persons:
        tag = "✓" if p["matched"] else "?"
        print(f"  {tag} {p['name']:28s}  energy {p['energy_score']:5.1f}  ({p['verdict']})")
    print(f"  report.html             → {REPORT_HTML_OUT}")
    print(f"  integration_output.json → {INTEGRATION_OUT}")
    print("=" * 55)