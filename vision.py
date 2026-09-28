"""
MILESTONE 1 - Visual Processing & Safety Detection
----------------------------------------------------
Detect PPE (helmet, vest, gloves...) in images and videos using YOLOv8.
"""
import os

import cv2
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")   # your trained PPE model


def load_model():
    """Load best.pt. Returns None if the file is missing."""
    if not os.path.exists(MODEL_PATH):
        return None
    return YOLO(MODEL_PATH)


def is_violation(class_name):
    """Any class starting with 'no' (no_helmet, NO-Hardhat...) is a violation."""
    name = class_name.lower()
    return name.startswith("no_") or name.startswith("no-") or name.startswith("no ")


def detect(model, image, conf=0.4):
    """Run YOLO on an image (file path or OpenCV frame)."""
    results = model.predict(image, conf=conf, verbose=False)

    detections = []
    for box in results[0].boxes:
        name = model.names[int(box.cls[0])]      # class names come from best.pt itself
        detections.append({
            "class": name,
            "conf": round(float(box.conf[0]), 2),
            "box": [int(v) for v in box.xyxy[0]],
            "violation": is_violation(name),
        })
    return detections


def draw(image, detections):
    """Red box = violation, green box = OK."""
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        color = (0, 0, 255) if d["violation"] else (0, 200, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{d['class']} {d['conf']}", (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image


def summarize(detections, source="image"):
    """Turn detections into simple text + numbers (the AI reads this text)."""
    counts = {}
    for d in detections:
        counts[d["class"]] = counts.get(d["class"], 0) + 1

    violations = [d["class"] for d in detections if d["violation"]]

    if violations:
        text = f"{source}: found {counts}. VIOLATIONS: {sorted(set(violations))}."
    elif detections:
        text = f"{source}: found {counts}. No PPE violations."
    else:
        text = f"{source}: nothing detected."

    return {"text": text, "counts": counts, "violations": violations, "compliant": not violations}


def detect_video(model, video_path, every_n=5, conf=0.4):
    """
    Check 1 out of every N frames (faster).
    The same worker appears in many frames, so for each class we keep the
    frame where it appeared the MOST times -> we count people, not frames.
    """
    cap = cv2.VideoCapture(video_path)
    best = {}              # class -> detections from its busiest frame
    preview = None         # frame with the most violations
    most_violations = -1
    frame_no = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        if frame_no % every_n != 0:
            continue

        dets = detect(model, frame, conf)

        per_class = {}
        for d in dets:
            per_class.setdefault(d["class"], []).append(d)
        for name, items in per_class.items():
            if len(items) > len(best.get(name, [])):
                best[name] = items

        n_viol = sum(d["violation"] for d in dets)
        if n_viol > most_violations:
            most_violations = n_viol
            preview = draw(frame.copy(), dets)

    cap.release()
    detections = [d for items in best.values() for d in items]
    return detections, preview, frame_no


# quick test:  python vision.py path/to/image.jpg
if __name__ == "__main__":
    import sys

    model = load_model()
    if model is None:
        print("best.pt not found")
    else:
        dets = detect(model, sys.argv[1])
        print(summarize(dets)["text"])
        cv2.imwrite("output.jpg", draw(cv2.imread(sys.argv[1]), dets))
        print("Saved output.jpg")
