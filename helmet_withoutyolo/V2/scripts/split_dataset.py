import cv2
import torch
import torchvision.transforms as T
from torchvision import models
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
import torch.nn as nn
from PIL import Image
import mediapipe as mp
import numpy as np

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =========================
# HELMET CLASSIFIER (your trained model)
# =========================
classifier = models.mobilenet_v2(weights=None)
classifier.classifier[1] = nn.Linear(classifier.last_channel, 2)

checkpoint = torch.load(
    r"D:\Dev\Coding\Safesight\helmet_withoutyolo\V2\scripts\helmet_model_best.pth",
    map_location=device
)
classifier.load_state_dict(checkpoint['model_state_dict'])
classifier.eval()
classifier = classifier.to(device)

class_names = checkpoint['class_names']
print(f"Classifier classes: {class_names}")

# =========================
# PERSON DETECTOR
# SSDLite MobileNetV3 — torchvision built-in, no downloads needed
# COCO class 1 = person
# This replaces HOG completely and is far more accurate
# =========================
print("Loading person detector (first run will download ~30MB)...")
detector_weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
person_detector = ssdlite320_mobilenet_v3_large(weights=detector_weights)
person_detector.eval()
person_detector = person_detector.to(device)
print("Person detector ready.")

# =========================
# TRANSFORMS
# =========================
classify_transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])

detector_transform = T.Compose([T.ToTensor()])

# =========================
# MEDIAPIPE — precise head localization
# =========================
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(min_detection_confidence=0.4)

# =========================
# VIDEO
# =========================
VIDEO_PATH = r"D:\Dev\Coding\Safesight\helmet_withoutyolo\V2\videoplayback.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Cannot open video")
    exit()

print("Video loaded. Press ESC to quit.\n")

# =========================
# CONFIG
# =========================
DETECT_EVERY_N = 3        # run SSD every 3 frames, track in between
CONFIDENCE_THRESHOLD = 0.78
PERSON_SCORE = 0.50       # SSD confidence to accept a person box
HISTORY_SIZE = 7
ALERT_THRESHOLD = 4       # violations in last 7 frames = alert

history = []
frame_count = 0
last_boxes = []


# =========================
# HELPERS
# =========================
def safe_crop(image, y1, y2, x1, x2):
    h, w = image.shape[:2]
    y1, y2 = max(0, int(y1)), min(h, int(y2))
    x1, x2 = max(0, int(x1)), min(w, int(x2))
    if y2 <= y1 or x2 <= x1:
        return None
    crop = image[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def classify_head(crop):
    if crop is None or crop.size == 0:
        return None, None
    if crop.shape[0] < 25 or crop.shape[1] < 25:
        return None, None
    try:
        img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        t = classify_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = classifier(t)
            probs = torch.softmax(out, dim=1)
            conf, pred = torch.max(probs, 1)
        return class_names[pred.item()], conf.item()
    except Exception:
        return None, None


def detect_persons(pil_image):
    t = detector_transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = person_detector(t)[0]
    boxes = []
    for box, label, score in zip(preds['boxes'], preds['labels'], preds['scores']):
        if label.item() == 1 and score.item() > PERSON_SCORE:
            x1, y1, x2, y2 = box.int().tolist()
            boxes.append((x1, y1, x2, y2))
    return boxes


# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    frame = cv2.resize(frame, (800, 600))
    fh, fw = frame.shape[:2]
    frame_count += 1

    # Run SSD every N frames for speed
    if frame_count % DETECT_EVERY_N == 1:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)
        last_boxes = detect_persons(pil_frame)

    violation_frame = 0

    for (px1, py1, px2, py2) in last_boxes:

        # Clip to frame
        px1, py1 = max(0, px1), max(0, py1)
        px2, py2 = min(fw, px2), min(fh, py2)
        pw, ph = px2 - px1, py2 - py1

        # Skip tiny/noise boxes
        if pw < 40 or ph < 60:
            continue

        person_roi = frame[py1:py2, px1:px2]
        roi_h, roi_w = person_roi.shape[:2]

        if person_roi.size == 0:
            continue

        # --------------------------------------------------
        # HEAD LOCALIZATION
        # 1. Try MediaPipe face inside person ROI
        # 2. Fallback: top 30% of person box
        # --------------------------------------------------
        crop = None
        hx1, hy1, hx2, hy2 = px1, py1, px2, py1 + int(ph * 0.30)

        try:
            roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
            face_results = face_detector.process(roi_rgb)

            if face_results.detections:
                det = face_results.detections[0]
                b = det.location_data.relative_bounding_box

                fx1 = int(b.xmin * roi_w)
                fy1 = int(b.ymin * roi_h)
                fx2 = fx1 + int(b.width * roi_w)
                fy2 = fy1 + int(b.height * roi_h)

                # Expand upward to capture helmet above face
                expand = int((fy2 - fy1) * 0.7)
                fy1 = max(0, fy1 - expand)

                candidate = safe_crop(person_roi, fy1, fy2, fx1, fx2)
                if candidate is not None:
                    crop = candidate
                    hx1 = px1 + fx1
                    hy1 = py1 + fy1
                    hx2 = px1 + fx2
                    hy2 = py1 + fy2

        except Exception:
            pass

        # Fallback
        if crop is None:
            crop = safe_crop(person_roi, 0, int(roi_h * 0.30), 0, roi_w)

        if crop is None:
            continue

        # --------------------------------------------------
        # CLASSIFY
        # --------------------------------------------------
        label, confidence = classify_head(crop)

        if label is None:
            continue

        # --------------------------------------------------
        # DRAW
        # --------------------------------------------------
        if label == "no_helmet" and confidence > CONFIDENCE_THRESHOLD:
            violation_frame += 1
            color = (0, 0, 255)
            txt = f"NO HELMET {confidence:.0%}"

            cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), color, 3)
            cv2.putText(frame, txt,
                        (px1, max(14, py1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        elif label == "helmet" and confidence > CONFIDENCE_THRESHOLD:
            color = (0, 200, 0)
            txt = f"HELMET {confidence:.0%}"

            cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), color, 2)
            cv2.putText(frame, txt,
                        (px1, max(14, py1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # --------------------------------------------------
    # TEMPORAL ALERT
    # --------------------------------------------------
    history.append(1 if violation_frame > 0 else 0)
    if len(history) > HISTORY_SIZE:
        history.pop(0)

    if sum(history) >= ALERT_THRESHOLD:
        cv2.rectangle(frame, (0, 0), (fw, 55), (0, 0, 160), -1)
        cv2.putText(frame, "ALERT: PERSON WITHOUT HELMET DETECTED",
                    (10, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (0, 0, 255), 2)

    cv2.putText(frame, f"Persons detected: {len(last_boxes)}  Violations: {violation_frame}",
                (10, fh - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("SafeSight - Helmet Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()