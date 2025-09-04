import cv2
import numpy as np
import grpc
from proto import vision_pb2
from proto import vision_pb2_grpc


def load_model():
    net = cv2.dnn.readNetFromCaffe(
        "models/MobileNetSSD_deploy.prototxt", "models/MobileNetSSD_deploy.caffemodel"
    )
    return net


def run_inference(frame, net):
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5
    )
    net.setInput(blob)
    detections = net.forward()
    h, w = frame.shape[:2]
    boxes = []
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > 0.5:
            idx = int(detections[0, 0, i, 1])
            label = str(idx)
            x_min = int(detections[0, 0, i, 3] * w)
            y_min = int(detections[0, 0, i, 4] * h)
            x_max = int(detections[0, 0, i, 5] * w)
            y_max = int(detections[0, 0, i, 6] * h)
            boxes.append((label, conf, x_min, y_min, x_max, y_max))
    return boxes


def main():
    net = load_model()
    channel = grpc.insecure_channel("localhost:50051")
    stub = vision_pb2_grpc.HapticFeedbackStub(channel)
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        boxes = run_inference(frame, net)
        message_boxes = []
        for b in boxes:
            message_boxes.append(
                vision_pb2.BoundingBox(
                    label=b[0],
                    confidence=b[1],
                    x_min=float(b[2]),
                    y_min=float(b[3]),
                    x_max=float(b[4]),
                    y_max=float(b[5]),
                )
            )
        request = vision_pb2.DetectionRequest(boxes=message_boxes)
        stub.SendDetection(request)
    cap.release()


if __name__ == "__main__":
    main()
