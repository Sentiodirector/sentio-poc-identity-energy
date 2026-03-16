# Named Face Identity + Energy Report
**Sentio Mind · POC Assignment · Project 1**

GitHub: https://github.com/Sentiodirector/sentio-poc-identity-energy.git
Branch: FirstName_LastName_RollNumber

---

## Why This Exists

Every person Sentio Mind detects in CCTV footage gets a random name. A school with 40 students ends up with 200 unnamed profiles the counsellor cannot use. This project fixes that. You get a folder of face photos with real names as filenames. You match the video to those photos, attach real names, compute energy levels, and output a clean report.

---

## What You Receive

```
p1_identity_energy/
├── known_faces/
│   ├── Arjun Mehta.jpg        ← one photo per person, filename = real name
│   ├── Priya Rajan.jpg
│   └── ...
├── video_sample_1.mov         ← download from dataset link in your assignment
├── identity_energy.py         ← your template — copy to solution.py
├── identity_energy.json       ← integration schema — do not change structure
└── README.md
```

---

## What You Must Build

Run `python solution.py` → it must produce:

1. `report.html` — one offline page: profile photo + name + energy bar + verdict per person
2. `integration_output.json` — follows `identity_energy.json` schema key for key

### Energy Formula (do not change)

```
energy_score = (face_brightness × 0.35) + (eye_openness × 0.30) + (movement_activity × 0.35)
```

- **face_brightness** — mean pixel brightness of face crop, 0 to 100
- **eye_openness** — eye height ÷ eye width from face mesh, 0 to 100
- **movement_activity** — optical flow magnitude in face region between frames, 0 to 100

Verdicts: ≥ 75 = High Energy · 50–74 = Moderate · below 50 = Low Energy

---

## Hard Rules

- Do not rename functions in `identity_energy.py`
- Do not change key names in `identity_energy.json`
- `report.html` must open from a local file with no internet
- Python 3.9+, no Jupyter notebooks

## Libraries

```
opencv-python==4.9.0   face_recognition==1.3.0   mediapipe==0.10.14
numpy==1.26.4          Pillow==10.3.0
```

---

## Submit

| # | File | What |
|---|------|------|
| 1 | `solution.py` | Working script |
| 2 | `report.html` | Generated energy report |
| 3 | `integration_output.json` | Output matching schema |
| 4 | `demo.mp4` | Screen recording under 2 min |

Push to your branch only. Do not touch main.

---

## Bonus

Detect unmatched faces in the video and list them as `UNKNOWN_001`, `UNKNOWN_002` in a separate section of the report.

*Sentio Mind · 2026*
