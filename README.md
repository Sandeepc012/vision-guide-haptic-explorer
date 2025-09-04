# VisionGuide Haptic Explorer (VGHE)

VisionGuide Haptic Explorer is a low‑latency AI system designed to assist visually impaired users through real‑time object detection and haptic feedback. The project builds a scalable edge computing framework that streams camera data, performs object recognition using a MobileNet SSD model, and communicates results to a haptic feedback device via gRPC.

## Features

- **Low‑Latency Object Detection:** Uses OpenCV's DNN module with a MobileNet SSD network for on‑device inference, configurable for INT8 quantized models for reduced latency.
- **gRPC Communication:** Encapsulates detection results as protocol buffer messages and streams them to a haptic feedback service.
- **Modular Design:** Separates the detection client and haptic server, enabling deployment on different devices and facilitating further integration of ASR and SLAM modules.

## Getting Started

Install dependencies listed in `requirements.txt` with pip:

```
pip install -r requirements.txt
```

Generate gRPC code from the protocol definitions:

```
python -m grpc_tools.protoc -I proto --python_out=proto --grpc_python_out=proto proto/vision.proto
```

Start the gRPC server:

```
python server.py
```

Run the detection client (ensure a webcam is attached or modify to read from a video file):

```
python client.py
```

## Model Weights

The detection client expects a MobileNet SSD Caffe model at `models/MobileNetSSD_deploy.prototxt` and `models/MobileNetSSD_deploy.caffemodel`. Download these files from the OpenCV model zoo and place them in a `models` directory relative to the repository root.
