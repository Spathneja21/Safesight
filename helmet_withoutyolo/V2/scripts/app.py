import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image
import mediapipe as mp
import numpy as np

# =========================
# MODEL
# =========================
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 2)

checkpoint = torch.load(
    r"D:\Dev\Coding\Safesight\helmet_withoutyolo\V2\scripts\helmet_model_best.pth",
    map_location="cpu"
)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
class_names = checkpoint['class_names']

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# DETECTORS
# =========================
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(min_detection_confidence=0.5)

# =========================
# VIDEO
# =========================
cap = cv2.VideoCapture(r"D:\Dev\Coding\Safesight\helmet_withoutyolo\V2\videoplayback.mp4")

history = []
HISTORY_SIZE = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (800,600))
    violation_frame = 0

    boxes, _ = hog.detectMultiScale(frame)

    for (x,y,w_box,h_box) in boxes:

        # 🔥 filter small detections
        if w_box < 80 or h_box < 120:
            continue

        person_roi = frame[y:y+h_box, x:x+w_box]
        if person_roi.size == 0:
            continue

        rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb)

        # =========================
        # CASE 1: FACE FOUND
        # =========================
        if results.detections:
            for det in results.detections:

                bbox = det.location_data.relative_bounding_box
                fx1 = int(bbox.xmin * w_box)
                fy1 = int(bbox.ymin * h_box)
                fw = int(bbox.width * w_box)
                fh = int(bbox.height * h_box)

                fx2 = fx1 + fw
                fy2 = fy1 + fh

                crop = person_roi[fy1:fy2, fx1:fx2]

                if crop.size == 0:
                    continue

                ch, cw = crop.shape[:2]

                # 🔥 reject weird crops
                if ch < 40 or cw < 40:
                    continue

                # 🔥 center crop only
                crop = crop[int(ch*0.2):int(ch*0.8),
                            int(cw*0.2):int(cw*0.8)]

        # =========================
        # CASE 2: NO FACE → fallback
        # =========================
        else:
            crop = person_roi[0:int(h_box * 0.25), :]

            ch, cw = crop.shape[:2]
            if ch < 40 or cw < 40:
                continue

        # =========================
        # COLOR FILTER (REMOVE HELMET OBJECT)
        # =========================
        mean_color = crop.mean(axis=(0,1))  # BGR

        if mean_color[2] > 150 and mean_color[2] > mean_color[1]:
            continue  # skip strong red objects

        # =========================
        # CLASSIFICATION
        # =========================
        img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        input_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)

        confidence = conf.item()
        label = class_names[pred.item()]

        # =========================
        # STRICT CONDITION
        # =========================
        if label == "no_helmet" and confidence > 0.8:
            violation_frame += 1

            cv2.rectangle(frame,
                          (x, y),
                          (x + w_box, y + int(h_box * 0.3)),
                          (0,0,255), 2)

            cv2.putText(frame, "NO HELMET!",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,0,255), 2)

    # =========================
    # TEMPORAL FILTER
    # =========================
    history.append(1 if violation_frame > 0 else 0)

    if len(history) > HISTORY_SIZE:
        history.pop(0)

    if sum(history) >= 3:
        cv2.putText(frame, "ALERT: NO HELMET!",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    3)

    cv2.imshow("Helmet Detection (Stable)", frame)

    if cv2.waitKey(25) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()