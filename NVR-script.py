import cv2
import requests
import time

# --- CONFIG ---
RTSP_URL = 'rtsp://your-camera-stream'
YOLO_CFG = 'yolov3.cfg'
YOLO_WEIGHTS = 'yolov3.weights'
COCO_NAMES = 'coco.names'
PUSHOVER_USER_KEY = 'your-user-key'
PUSHOVER_API_TOKEN = 'your-api-token'
NOTIFY_INTERVAL = 60  # seconds between notifications
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

# --- LOAD LABELS ---
with open(COCO_NAMES, 'r') as f:
    labels = f.read().strip().split('\n')

# --- LOAD YOLO ---
net = cv2.dnn.readNetFromDarknet(YOLO_CFG, YOLO_WEIGHTS)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# --- VIDEO STREAM ---
cap = cv2.VideoCapture(RTSP_URL)
last_notified = 0

def send_pushover_notification():
    print("[+] Human detected, sending notification...")
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": PUSHOVER_API_TOKEN,
            "user": PUSHOVER_USER_KEY,
            "message": "🚨 Human detected on RTSP camera!",
        }
    )

while True:
    ret, frame = cap.read()
    if not ret:
        print("[-] Failed to grab frame.")
        break

    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for out in outputs:
        for detection in out:
            scores = detection[5:]
            class_id = int(scores.argmax())
            confidence = scores[class_id]

            if confidence > CONFIDENCE_THRESHOLD and labels[class_id] == 'person':
                center_x, center_y, w, h = (detection[0:4] * [width, height, width, height]).astype('int')
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    if len(indices) > 0 and (time.time() - last_notified > NOTIFY_INTERVAL):
        send_pushover_notification()
        last_notified = time.time()

    # Optional: Show frame for debugging
    # for i in indices.flatten():
    #     x, y, w, h = boxes[i]
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.imshow("Frame", frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
cv2.destroyAllWindows()
