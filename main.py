import cv2
from ultralytics import YOLO

model = YOLO("best.pt")  

ROI_TOP_LEFT = (100, 100)
ROI_BOTTOM_RIGHT = (400, 400)

cap = cv2.VideoCapture(0) 

if not cap.isOpened():
    print("Could not open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1) 

    x1, y1 = ROI_TOP_LEFT
    x2, y2 = ROI_BOTTOM_RIGHT

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    roi = frame[y1:y2, x1:x2]
    results = model(roi, verbose=False)

    pred = results[0]
    class_id = pred.probs.top1
    conf = pred.probs.top1conf.item()
    label = model.names[class_id]

    text = f"{label} ({conf*100:.1f}%)"
    cv2.putText(frame, text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("ASL Hand Sign Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()