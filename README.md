# 👁️ AI Vision Counter

> A real-time AI-powered human detection, instance segmentation, tracking, and bidirectional people counting system.

AI Vision Counter is a computer vision application designed to detect and segment **humans only** from live video streams, track their movement, and accurately count people entering and exiting a defined boundary.

The project combines **YOLO Instance Segmentation**, object tracking, boundary-crossing logic, and a modern real-time dashboard to simulate practical use cases such as retail stores, shopping malls, offices, and public spaces.

---

## 🚀 Live Demo

🔗 **Live Application:** Coming Soon

> Open the application on a laptop to use the webcam dashboard.  
> Mobile devices are automatically optimized for phone camera streaming.

---

## ✨ Key Features

### 🤖 AI-Powered Human Detection
- Detects **persons only**
- Ignores non-human objects such as phones, earphones, bags, etc.
- Uses a single YOLO segmentation model
- Real-time instance segmentation

### 🎭 Instance Segmentation
Unlike traditional bounding-box detection, the system creates precise masks around detected people.

- Per-person segmentation masks
- Real-time visualization
- Human-only filtering

### 👥 Multi-Person Tracking
- Persistent person IDs
- Real-time movement tracking
- Helps prevent duplicate counting

### ↔️ Bidirectional People Counting
The system tracks people crossing a configurable boundary line.

- **IN Count** — People entering
- **OUT Count** — People exiting
- **Live Count** — Current people count
- Recent crossing events

### 🛡️ Boundary Stability / Dead Zone
A configurable safety zone around the boundary prevents false counts caused by:

- Tracking jitter
- Small movements near the boundary
- Repeated oscillation
- Unstable detections

A person must clearly move from one stable side of the boundary to the other before a crossing is counted.

### 📷 Multiple Camera Workflows

#### 💻 Webcam
Use the computer's built-in or connected webcam directly from the dashboard.

#### 📱 Phone Camera
Use a mobile phone as a remote live camera.

- QR code connection
- Rear camera preferred by default
- Front/back camera switching
- Responsive mobile camera interface
- Live frame streaming

#### 📹 CCTV — Coming Soon
Future support for RTSP/IP camera streams.

#### 🎬 Video File — Coming Soon
Future support for uploading and analyzing recorded videos.

---

## 🖥️ Dashboard Features

- Live AI video stream
- Human instance segmentation
- Real-time people counting
- IN / OUT statistics
- Live count monitoring
- Recent crossing events
- Adjustable boundary line
- Confidence threshold control
- Boundary stability control
- Mask visualization toggle
- Tracking visualization
- FPS monitoring
- Processing latency display
- Reset counter functionality

---

## 🏗️ System Architecture

```text
                    ┌─────────────────┐
                    │  Video Sources  │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
         💻 Webcam      📱 Phone Camera    Future CCTV
              │              │
              └──────────────┼──────────────┘
                             │
                             ▼
                   ┌──────────────────┐
                   │ Frame Processing │
                   └────────┬─────────┘
                             │
                             ▼
                  ┌────────────────────┐
                  │ YOLO Segmentation  │
                  │   Person Class     │
                  └─────────┬──────────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │ Multi-Object Track │
                  └─────────┬──────────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │ Boundary Crossing  │
                  │    + Dead Zone     │
                  └─────────┬──────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
           IN Count      OUT Count    Live Count