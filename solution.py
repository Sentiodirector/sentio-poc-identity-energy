import cv2
import json
import base64
import time
import numpy as np
from pathlib import Path
from datetime import date
import face_recognition
import mediapipe as mp
import warnings
warnings.filterwarnings('ignore')

# CONFIG
KNOWN_FACES_DIR  = Path("known_faces")
VIDEO_PATH       = Path("Class_8_cctv_video_1.mov")
REPORT_HTML_OUT  = Path("report.html")
INTEGRATION_OUT  = Path("integration_output.json")

SCHOOL_NAME      = "Demo School"
MATCH_THRESHOLD  = 0.55
MAX_KEYFRAMES    = 20

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)
unknown_encodings = []


# HELPERS
def encode_b64(img, size=(240, 240)):
    if img is None or img.size == 0:
        return ""
    img = cv2.resize(img, size)
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf).decode("utf-8")


def verdict(score):
    return "high" if score >= 75 else "moderate" if score >= 50 else "low"


def is_valid_face(face):
    try:
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        res = mp_face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            return False

        lm = res.multi_face_landmarks[0].landmark
        if abs(lm[33].x - lm[263].x) < 0.05:
            return False

        return True
    except:
        return False


# STEP 1
def load_known_faces(folder):
    known = {}
    for path in folder.glob("*"):
        img = face_recognition.load_image_file(path)
        enc = face_recognition.face_encodings(img)
        if enc:
            known.setdefault(path.stem, []).extend(enc)
    return known


# STEP 2
def extract_keyframes(video_path, max_frames):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // max_frames)

    frames = []
    i = 0

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if i % step == 0:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.createCLAHE(2.0, (8, 8)).apply(l)
            frame = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
            frames.append((i, frame))

        i += 1

    cap.release()
    return frames


# STEP 3
def detect_and_match(frame, known, threshold):
    global unknown_encodings

    detections = []
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    locs = face_recognition.face_locations(rgb)
    encs = face_recognition.face_encodings(rgb, locs)

    for (t, r, b, l), enc in zip(locs, encs):
        name = "UNKNOWN"
        matched = False
        conf = 0.0
        # match known
        best_name, best_dist = None, 1.0
        for k, v in known.items():
            d = np.min(face_recognition.face_distance(v, enc))
            if d < best_dist:
                best_dist, best_name = d, k

        if best_dist < threshold:
            conf = 1 - best_dist

            if conf > 0.5:
                name, matched = best_name, True
            else:
                matched, conf = False, 0.0

        elif best_dist < (threshold + 0.02):   
            name, matched = best_name, True
            conf = 1 - best_dist
        else:
            matched, conf = False, 0.0
            bbox = (l, t, r - l, b - t)

            assigned = False
            for i, (u_enc, u_box) in enumerate(unknown_encodings):
                d = face_recognition.face_distance([u_enc], enc)[0]

                cx1 = u_box[0] + u_box[2]//2
                cy1 = u_box[1] + u_box[3]//2
                cx2 = bbox[0] + bbox[2]//2
                cy2 = bbox[1] + bbox[3]//2

                spatial = np.hypot(cx1 - cx2, cy1 - cy2)

                if d < 0.6 and spatial < 100:
                    name = f"UNKNOWN_{i+1:03d}"
                    unknown_encodings[i] = (0.7*u_enc + 0.3*enc, bbox)
                    assigned = True
                    break

            if not assigned:
                unknown_encodings.append((enc, bbox))
                name = f"UNKNOWN_{len(unknown_encodings):03d}"

        face = frame[t:b, l:r]
        if face.size == 0:
            continue

        h, w = face.shape[:2]
        if h < 20 or w < 20:
            continue

        if h < 50 or w < 50:
            face = cv2.resize(face, (80, 80))

        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        if np.var(gray) < 100:
            continue
        if not (40 < np.mean(gray) < 220):
            continue
        if not (0.6 < w / float(h) < 1.6):
            continue
        if not is_valid_face(face):
            continue

        detections.append({
            "name": name,
            "matched": matched,
            "confidence": round(conf, 3),
            "bbox": (l, t, r-l, b-t),
            "face_crop": face
        })

    return detections


# STEP 4
def compute_face_brightness(face):
    return np.clip(np.mean(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY))/2.55 * 1.2, 0, 100)


def compute_eye_openness(face):
    try:
        res = mp_face_mesh.process(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return 50

        lm = res.multi_face_landmarks[0].landmark

        def d(a,b): return np.linalg.norm([a.x-b.x, a.y-b.y])

        val = (d(lm[159],lm[145])/d(lm[33],lm[133]) +
               d(lm[386],lm[374])/d(lm[362],lm[263]))/2

        return np.clip(val*350,0,100)
    except:
        return 50


def compute_movement(prev, curr, bbox):
    if prev is None:
        return 0.0

    x,y,w,h = bbox
    p = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)[y:y+h, x:x+w]
    c = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)[y:y+h, x:x+w]

    if p.shape != c.shape or p.size == 0:
        return 0.0

    flow = cv2.calcOpticalFlowFarneback(p, c, None, 0.5,3,15,3,5,1.2,0)
    mag,_ = cv2.cartToPolar(flow[...,0], flow[...,1])

    return np.clip(np.mean(mag)*25,0,100)


# STEP 5
def aggregate_persons(all_detections):
    groups = {}
    for d in all_detections:
        groups.setdefault(d["name"], []).append(d)

    persons = []
    pid = 1

    for name, dets in groups.items():

        if len(dets) < 1:
            continue

        matched = sum(d["matched"] for d in dets) > len(dets)/2

        names = [d["name"] for d in dets if d["matched"]]
        if names:
            name = max(set(names), key=names.count)

        b = np.mean([d["brightness"] for d in dets])
        e = np.mean([d["eye_openness"] for d in dets])
        m = np.mean([d["movement"] for d in dets])

        score = 0.35*b + 0.30*e + 0.35*m

        def pick(d):
            sharp = cv2.Laplacian(cv2.cvtColor(d["face_crop"], cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            return 0.6*d["confidence"] + 0.4*(sharp/1000)

        best = max(dets, key=pick)["face_crop"]

        persons.append({
            "person_id": f"SCHOOL_P{pid:04d}",
            "name": name,
            "matched": matched,
            "match_confidence": np.mean([d["confidence"] for d in dets]),
            "profile_image_b64": encode_b64(best),
            "frames_detected": len(dets),
            "energy_score": round(score,1),
            "energy_breakdown": {
                "face_brightness": int(b),
                "eye_openness": int(e),
                "movement_activity": int(m)
            },
            "verdict": verdict(score),
            "first_seen_frame": min(d["frame_idx"] for d in dets),
            "last_seen_frame": max(d["frame_idx"] for d in dets)
        })

        pid += 1

    u = 1
    for p in persons:
        if not p["matched"]:
            p["name"] = f"UNKNOWN_{u:03d}"
            u += 1

    return persons


# STEP 6
def generate_report(persons, output_path):
    known = [p for p in persons if p["matched"]]
    unknown = [p for p in persons if not p["matched"]]

    def render(data):
        html = ""
        for p in data:
            color = "#2ecc71" if p["verdict"]=="high" else "#f39c12" if p["verdict"]=="moderate" else "#e74c3c"
            html += f"""
            <div class="card">
                <img src="data:image/jpeg;base64,{p['profile_image_b64']}">
                <div class="content">
                    <h3>{p['name']}</h3>
                    <p>Energy: {p['energy_score']}</p>
                    <div class="bar"><div class="fill" style="width:{p['energy_score']}%;background:{color};"></div></div>
                    <p>{p['verdict']}</p>
                </div>
            </div>
            """
        return html

    html = f"""
    <html><head><style>
    body {{font-family:Arial;background:#eef2f7;padding:20px;}}
    .container {{display:flex;flex-wrap:wrap;}}
    .card {{width:240px;margin:10px;background:white;border-radius:10px;box-shadow:0 3px 8px rgba(0,0,0,0.1);}}
    img {{width:100%;height:220px;object-fit:cover;}}
    .content {{padding:10px;}}
    .bar {{height:8px;background:#ddd;border-radius:4px;}}
    .fill {{height:8px;border-radius:4px;}}
    </style></head><body>

    <h1>Student Energy Report</h1>

    <h2>Known Students</h2>
    <div class="container">{render(known)}</div>

    <h2>Unknown Students</h2>
    <div class="container">{render(unknown)}</div>

    </body></html>
    """

    open(output_path,"w").write(html)


# STEP 7
def write_integration_json(persons, output_path, video_name, processing_time):
    output = {
        "source": "p1_identity_energy",
        "school": SCHOOL_NAME,
        "date": str(date.today()),
        "video_file": video_name,
        "total_persons_matched": sum(p["matched"] for p in persons),
        "total_persons_unknown": sum(not p["matched"] for p in persons),
        "processing_time_sec": processing_time,
        "persons": persons,
    }
    json.dump(output, open(output_path,"w"), indent=2)


# MAIN
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