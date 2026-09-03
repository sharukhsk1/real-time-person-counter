"""
Vision Counter V3
=================
Single-model, real-time human instance segmentation + robust bidirectional
line-crossing counter.

Model: yolov8n-seg.pt ONLY
Detection class: COCO class 0 (person) ONLY
Tracking: ByteTrack through Ultralytics

Sources supported:
- Webcam: 0, 1, ...
- IP camera / phone camera URL (MJPEG/HTTP)
- RTSP CCTV stream
- Video file path

The counter intentionally uses a strict state machine so a jittering mask,
hand/object movement, or a sudden tracker jump is much less likely to create
a false IN/OUT event.
"""

import asyncio
import base64
import json
import logging
import os
import platform
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vision_counter")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CounterConfig:
    # SINGLE MODEL ONLY
    model_path: str = "yolov8n-seg.pt"
    tracker: str = "bytetrack.yaml"

    # YOLO
    conf_threshold: float = 0.45
    iou_threshold: float = 0.50
    target_classes: List[int] = field(default_factory=lambda: [0])  # PERSON ONLY

    # Processing
    resize_width: int = 960
    max_processing_fps: int = 25
    jpeg_quality: int = 82

    # Boundary
    line_position_ratio: float = 0.50
    hysteresis_pixels: int = 45

    # Strict human / track validation
    min_track_age: int = 8
    min_person_width: int = 30
    min_person_height: int = 70
    min_box_area: int = 3000

    # Crossing validation
    min_side_frames: int = 4
    min_crossing_distance: int = 75
    max_anchor_jump: int = 140
    crossing_cooldown: float = 1.25

    # Tracking cleanup
    stale_track_seconds: float = 2.5

    # Visualization
    mask_alpha: float = 0.30
    show_masks: bool = True
    show_track_ids: bool = True


# =============================================================================
# MODEL
# =============================================================================

def resolve_model(model_path: str) -> str:
    candidates = [
        Path(model_path),
        Path(__file__).resolve().parent / model_path,
        Path.home() / ".ultralytics" / "weights" / model_path,
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    raise FileNotFoundError(
        f"Model '{model_path}' was not found. Put yolov8n-seg.pt "
        f"in the same folder as app.py."
    )


# =============================================================================
# TRACK STATE
# =============================================================================

class Zone:
    UNKNOWN = "unknown"
    LEFT = "left"
    RIGHT = "right"


@dataclass
class PersonTrack:
    track_id: int

    # Stable side state
    stable_zone: str = Zone.UNKNOWN

    # Candidate side observations
    left_frames: int = 0
    right_frames: int = 0

    # Crossing state
    crossing_from: Optional[str] = None
    crossing_start_x: Optional[int] = None
    last_cross_time: float = 0.0

    # Detection history
    age: int = 0
    last_seen: float = 0.0
    previous_anchor: Optional[Tuple[int, int]] = None
    anchors: Deque[Tuple[int, int]] = field(
        default_factory=lambda: deque(maxlen=8)
    )

    # Visual data
    bbox: Optional[Tuple[int, int, int, int]] = None
    mask_points: Optional[np.ndarray] = None
    confidence: float = 0.0

    color: Tuple[int, int, int] = field(
        default_factory=lambda: (
            int(np.random.randint(70, 220)),
            int(np.random.randint(70, 220)),
            int(np.random.randint(70, 220)),
        )
    )


# =============================================================================
# ROBUST BIDIRECTIONAL COUNTER
# =============================================================================

class BidirectionalLineCounter:
    """
    Strict crossing state machine:

        STABLE LEFT
            |
            v
        DEAD ZONE
            |
            v
        STABLE RIGHT for N frames + minimum travel
            => IN

    Reverse direction => OUT.

    Merely jittering around the line does not count.
    A sudden tracker teleport is rejected.
    Tiny detections are rejected.
    Only YOLO class 0 (person) reaches this logic.
    """

    def __init__(self, config: CounterConfig, frame_width: int, frame_height: int):
        self.config = config
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.line_x = int(frame_width * config.line_position_ratio)

        self.tracks: Dict[int, PersonTrack] = {}

        self.total_count = 0
        self.in_count = 0
        self.out_count = 0

        self.events = deque(maxlen=50)
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = time.time()

    def set_line_ratio(self, ratio: float):
        ratio = max(0.05, min(0.95, float(ratio)))
        self.config.line_position_ratio = ratio
        self.line_x = int(self.frame_width * ratio)

    def reset(self):
        """Reset counts and per-track crossing state safely."""
        self.total_count = 0
        self.in_count = 0
        self.out_count = 0
        self.events.clear()

        # Clear tracks completely so no old crossing history can
        # generate a false event immediately after reset.
        self.tracks.clear()
        self.fps_history.clear()
        self.last_frame_time = time.time()

        logger.info("Counter reset")

    def _get_zone(self, x: int) -> Optional[str]:
        dead = self.config.hysteresis_pixels

        if x < self.line_x - dead:
            return Zone.LEFT
        if x > self.line_x + dead:
            return Zone.RIGHT

        return None

    @staticmethod
    def _foot_anchor(box) -> Tuple[int, int]:
        """
        Bottom-center is more suitable for CCTV people counting than the
        mask centroid. Arms/hands/object movement can distort a centroid,
        while the bottom-center usually represents the person's ground path.
        """
        x1, y1, x2, y2 = map(int, box)
        return int((x1 + x2) / 2), int(y2)

    def _add_event(self, direction: str, track_id: int, x: int, y: int):
        self.events.append(
            {
                "time": time.strftime("%H:%M:%S"),
                "direction": direction,
                "track_id": int(track_id),
                "position": {"x": int(x), "y": int(y)},
                "current_count": int(self.total_count),
            }
        )

    def _register_crossing(
        self,
        track: PersonTrack,
        direction: str,
        anchor: Tuple[int, int],
        now: float,
    ):
        if now - track.last_cross_time < self.config.crossing_cooldown:
            return

        if direction == "IN":
            self.total_count += 1
            self.in_count += 1
        else:
            self.total_count = max(0, self.total_count - 1)
            self.out_count += 1

        track.last_cross_time = now
        track.crossing_from = None
        track.crossing_start_x = None

        self._add_event(direction, track.track_id, anchor[0], anchor[1])

        logger.info(
            "%s | person=%s | occupancy=%s",
            direction,
            track.track_id,
            self.total_count,
        )

    def _update_crossing(
        self,
        track: PersonTrack,
        anchor: Tuple[int, int],
        now: float,
    ):
        """
        Update strict per-person crossing state.

        Important:
        - Stable zone must be observed for several frames.
        - Entering the dead zone arms a possible crossing.
        - Opposite side must be stable for several frames.
        - Travel distance must exceed minimum.
        """
        x, _ = anchor
        zone = self._get_zone(x)

        # ---------------------------------------------------------------------
        # INSIDE DEAD ZONE: arm a crossing, but NEVER count here.
        # ---------------------------------------------------------------------
        if zone is None:
            if (
                track.stable_zone in (Zone.LEFT, Zone.RIGHT)
                and track.crossing_from is None
            ):
                track.crossing_from = track.stable_zone
                track.crossing_start_x = x

            track.left_frames = 0
            track.right_frames = 0
            return

        # ---------------------------------------------------------------------
        # COUNT STABLE SIDE OBSERVATIONS
        # ---------------------------------------------------------------------
        if zone == Zone.LEFT:
            track.left_frames += 1
            track.right_frames = 0
            stable_frames = track.left_frames
        else:
            track.right_frames += 1
            track.left_frames = 0
            stable_frames = track.right_frames

        # New tracks need a stable initial side, never count on initialization.
        if track.stable_zone == Zone.UNKNOWN:
            if stable_frames >= self.config.min_side_frames:
                track.stable_zone = zone
                track.crossing_from = None
                track.crossing_start_x = None
            return

        # Same side again: cancel an unfinished crossing attempt.
        if zone == track.stable_zone:
            if stable_frames >= self.config.min_side_frames:
                track.crossing_from = None
                track.crossing_start_x = None
            return

        # Opposite side must be stable before anything can count.
        if stable_frames < self.config.min_side_frames:
            return

        # ---------------------------------------------------------------------
        # REAL CROSSING VALIDATION
        # ---------------------------------------------------------------------
        if (
            track.crossing_from is not None
            and track.crossing_from != zone
            and track.crossing_start_x is not None
        ):
            distance = abs(x - track.crossing_start_x)

            if distance >= self.config.min_crossing_distance:
                if (
                    track.crossing_from == Zone.LEFT
                    and zone == Zone.RIGHT
                ):
                    self._register_crossing(track, "IN", anchor, now)

                elif (
                    track.crossing_from == Zone.RIGHT
                    and zone == Zone.LEFT
                ):
                    self._register_crossing(track, "OUT", anchor, now)

        # Opposite side becomes the new stable state.
        track.stable_zone = zone

    def update(self, results, frame: np.ndarray) -> np.ndarray:
        now = time.time()

        dt = now - self.last_frame_time
        self.last_frame_time = now
        if dt > 0:
            self.fps_history.append(1.0 / dt)

        active_ids = set()

        if (
            results
            and results[0].boxes is not None
            and results[0].boxes.id is not None
        ):
            result = results[0]

            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.int().cpu().numpy()

            masks = None
            if result.masks is not None and hasattr(result.masks, "xy"):
                masks = result.masks.xy

            for i, (box, track_id, conf, cls) in enumerate(
                zip(boxes, track_ids, confidences, classes)
            ):
                # Absolute guarantee: PERSON CLASS ONLY.
                if int(cls) != 0:
                    continue

                if float(conf) < self.config.conf_threshold:
                    continue

                x1, y1, x2, y2 = map(int, box)
                bw = max(0, x2 - x1)
                bh = max(0, y2 - y1)
                area = bw * bh

                # Reject very small / weak person detections.
                if (
                    bw < self.config.min_person_width
                    or bh < self.config.min_person_height
                    or area < self.config.min_box_area
                ):
                    continue

                tid = int(track_id)
                active_ids.add(tid)

                if tid not in self.tracks:
                    self.tracks[tid] = PersonTrack(track_id=tid)

                track = self.tracks[tid]
                track.age += 1
                track.last_seen = now
                track.confidence = float(conf)
                track.bbox = (x1, y1, x2, y2)

                if masks is not None and i < len(masks):
                    polygon = np.asarray(masks[i])
                    track.mask_points = (
                        polygon if len(polygon) >= 3 else None
                    )

                # Bottom-center anchor: robust for line crossing.
                anchor = self._foot_anchor(box)

                # Reject unrealistic tracker jumps.
                if track.previous_anchor is not None:
                    px, py = track.previous_anchor
                    jump = float(
                        np.hypot(anchor[0] - px, anchor[1] - py)
                    )

                    if jump > self.config.max_anchor_jump:
                        # Update visual position, but do not use this frame
                        # to change the crossing state.
                        track.previous_anchor = anchor
                        track.anchors.clear()
                        track.anchors.append(anchor)
                        continue

                track.previous_anchor = anchor
                track.anchors.append(anchor)

                # Young tracks are displayed but cannot count yet.
                if track.age < self.config.min_track_age:
                    continue

                self._update_crossing(track, anchor, now)

        # Remove stale tracks.
        stale = [
            tid
            for tid, track in self.tracks.items()
            if tid not in active_ids
            and (now - track.last_seen) > self.config.stale_track_seconds
        ]
        for tid in stale:
            del self.tracks[tid]

        return self._draw_annotations(frame.copy())

    def _draw_annotations(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]

        # ---------------------------------------------------------------------
        # SEGMENTATION MASKS
        # ---------------------------------------------------------------------
        if self.config.show_masks:
            overlay = frame.copy()

            for track in self.tracks.values():
                if (
                    track.mask_points is not None
                    and len(track.mask_points) >= 3
                ):
                    pts = (
                        track.mask_points
                        .reshape((-1, 1, 2))
                        .astype(np.int32)
                    )

                    cv2.fillPoly(overlay, [pts], track.color)
                    cv2.polylines(frame, [pts], True, track.color, 2)

            cv2.addWeighted(
                overlay,
                self.config.mask_alpha,
                frame,
                1 - self.config.mask_alpha,
                0,
                frame,
            )

        # ---------------------------------------------------------------------
        # TRACK LABELS + TRAILS
        # ---------------------------------------------------------------------
        for track in self.tracks.values():
            if track.bbox is None:
                continue

            x1, y1, x2, y2 = track.bbox

            # Trail
            if len(track.anchors) >= 2:
                points = list(track.anchors)
                for j in range(1, len(points)):
                    cv2.line(
                        frame,
                        points[j - 1],
                        points[j],
                        track.color,
                        2,
                    )

            # Bottom-center anchor
            if track.previous_anchor is not None:
                ax, ay = track.previous_anchor
                cv2.circle(frame, (ax, ay), 5, (255, 255, 255), -1)
                cv2.circle(frame, (ax, ay), 3, track.color, -1)

            if self.config.show_track_ids:
                label = f"PERSON #{track.track_id}"
                (tw, th), _ = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    2,
                )

                label_y = max(th + 10, y1 - 6)
                cv2.rectangle(
                    frame,
                    (x1, label_y - th - 10),
                    (x1 + tw + 12, label_y + 2),
                    track.color,
                    -1,
                )
                cv2.putText(
                    frame,
                    label,
                    (x1 + 6, label_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        # ---------------------------------------------------------------------
        # BOUNDARY + DEAD ZONE
        # ---------------------------------------------------------------------
        dead = self.config.hysteresis_pixels
        line_x = self.line_x

        # Very subtle dead-zone overlay.
        overlay = frame.copy()
        left_dead = max(0, line_x - dead)
        right_dead = min(w, line_x + dead)
        cv2.rectangle(
            overlay,
            (left_dead, 0),
            (right_dead, h),
            (0, 255, 255),
            -1,
        )
        cv2.addWeighted(overlay, 0.06, frame, 0.94, 0, frame)

        # Glow
        for thickness, alpha in [(16, 0.10), (10, 0.18), (6, 0.30)]:
            glow = frame.copy()
            cv2.line(
                glow,
                (line_x, 0),
                (line_x, h),
                (0, 255, 255),
                thickness,
            )
            cv2.addWeighted(glow, alpha, frame, 1 - alpha, 0, frame)

        # Main boundary
        cv2.line(
            frame,
            (line_x, 0),
            (line_x, h),
            (0, 255, 255),
            3,
            cv2.LINE_AA,
        )

        # Zone labels
        cv2.putText(
            frame,
            "ZONE A",
            (max(25, line_x - 170), 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (210, 210, 210),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "ZONE B",
            (min(w - 150, line_x + 45), 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (210, 210, 210),
            2,
            cv2.LINE_AA,
        )

        # Direction arrows
        mid_y = h // 2
        cv2.arrowedLine(
            frame,
            (max(20, line_x - 120), mid_y - 35),
            (line_x - 15, mid_y - 35),
            (0, 230, 80),
            3,
            cv2.LINE_AA,
            tipLength=0.25,
        )
        cv2.arrowedLine(
            frame,
            (min(w - 20, line_x + 120), mid_y + 35),
            (line_x + 15, mid_y + 35),
            (40, 70, 255),
            3,
            cv2.LINE_AA,
            tipLength=0.25,
        )

        # Small boundary pill
        pill = "BOUNDARY"
        (tw, th), _ = cv2.getTextSize(
            pill,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2,
        )
        px = max(5, min(w - tw - 30, line_x - tw // 2 - 15))
        cv2.rectangle(
            frame,
            (px, 12),
            (px + tw + 30, 48),
            (20, 45, 50),
            -1,
        )
        cv2.putText(
            frame,
            pill,
            (px + 15, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return frame

    def get_stats(self) -> dict:
        fps = (
            float(np.mean(self.fps_history))
            if self.fps_history
            else 0.0
        )

        return {
            "total_count": int(self.total_count),
            "in_count": int(self.in_count),
            "out_count": int(self.out_count),
            "active_tracks": len(self.tracks),
            "fps": round(fps, 1),
            "events": list(self.events)[-10:],
            "line_position": round(
                self.config.line_position_ratio * 100
            ),
            "model": "YOLOv8n-Seg",
            "person_only": True,
        }


# =============================================================================
# VIDEO PROCESSOR
# =============================================================================

class VideoProcessor:
    def __init__(self, config: CounterConfig):
        self.config = config

        self.model = None
        self.model_loaded = False
        self.model_error: Optional[str] = None

        self.cap: Optional[cv2.VideoCapture] = None
        self.counter: Optional[BidirectionalLineCounter] = None

        self.is_running = False
        self.processing_task: Optional[asyncio.Task] = None

        self.processed_frame: Optional[np.ndarray] = None

        # Phone camera source. Frames arrive from /phone-stream over WebSocket.
        # The same VideoProcessor and the same single YOLO model process them.
        self.phone_frame: Optional[np.ndarray] = None
        self.phone_frame_seq = 0
        self.phone_processed_seq = -1
        self.phone_connected = False
        self.source_mode = "capture"

        self.lock = threading.Lock()
        self.last_source = "0"

    def load_model(self):
        if self.model_loaded and self.model is not None:
            return

        from ultralytics import YOLO

        model_path = resolve_model(self.config.model_path)
        logger.info("Loading SINGLE model: %s", model_path)

        self.model = YOLO(model_path)
        self.model_loaded = True
        self.model_error = None

        logger.info("YOLO segmentation model loaded successfully")

    def _open_capture(self, source: str):
        """Open a server-side video source safely.

        Important deployment behaviour:
        A Render/Railway/container server does not have access to the webcam
        attached to the visitor's laptop. Numeric sources (0, 1, ...) are
        therefore valid only when a physical camera is attached to the machine
        running this Python process. Browser/phone cameras must send frames via
        /phone-stream.
        """
        if source is None or str(source).strip() == "":
            source = "0"

        source = str(source).strip()
        actual_source = int(source) if source.isdigit() else source

        # Stop old camera safely.
        self.stop_capture()

        # Avoid noisy OpenCV/FFMPEG warnings on cloud Linux containers that
        # have no /dev/video devices at all.
        if isinstance(actual_source, int) and not os_name_is_windows():
            camera_path = Path(f"/dev/video{actual_source}")
            if not camera_path.exists():
                raise RuntimeError(
                    f"Server camera index {actual_source} is unavailable on this deployment. "
                    "A deployed server cannot access your laptop webcam directly. "
                    "Use Phone Camera (QR) for browser/mobile streaming, or provide an RTSP/IP camera URL or video file."
                )

        if isinstance(actual_source, int) and os_name_is_windows():
            cap = cv2.VideoCapture(actual_source, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(actual_source)

        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                f"Cannot open source: {source}. "
                "Check webcam index, phone/IP camera URL, RTSP URL, or CCTV stream."
            )

        # Low-latency settings where backend supports them.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self.cap = cap
        self.last_source = source

        # Read one frame so dimensions are real.
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.stop_capture()
            raise RuntimeError(f"Source opened but no frame was received: {source}")

        frame = self._resize_frame(frame)
        fh, fw = frame.shape[:2]

        self.counter = BidirectionalLineCounter(
            self.config,
            fw,
            fh,
        )

        self.is_running = True

        with self.lock:
            self.processed_frame = frame.copy()

        logger.info("Source started: %s (%sx%s)", source, fw, fh)

    def start_browser_capture(self, mode: str = "phone"):
        """
        Prepare the existing single YOLO processor for browser camera frames.
    
        Supported modes:
        - phone
        - browser_webcam
    
        Frames arrive through WebSocket.
        No server-side cv2.VideoCapture() is used.
        """
    
        mode = (
            "browser_webcam"
            if mode == "browser_webcam"
            else "phone"
        )
    
        # Stop previous server-side capture safely.
        self.stop_capture()
    
        self.source_mode = mode
    
        # Reset incoming browser frame state.
        self.phone_frame = None
        self.phone_frame_seq = 0
        self.phone_processed_seq = -1
    
        # Browser camera can have a different resolution.
        # Recreate the counter when the first frame arrives.
        self.counter = None
    
        with self.lock:
            self.processed_frame = None
    
        self.phone_connected = True
        self.is_running = True
    
        if mode == "browser_webcam":
            self.last_source = "BROWSER_WEBCAM"
        else:
            self.last_source = "PHONE_CAMERA"
    
        logger.info(
            "Browser camera source prepared: %s",
            mode
        )
    
    
    def start_phone_capture(self):
        """Prepare phone browser camera streaming."""
        self.start_browser_capture("phone")
    
    
    def start_webcam_capture(self):
        """Prepare desktop browser webcam streaming."""
        self.start_browser_capture("browser_webcam")

    def submit_phone_jpeg(self, jpeg_bytes: bytes) -> bool:
        """
        Receive one JPEG frame from a browser camera.
    
        Used by:
        - Phone camera
        - Desktop browser webcam
    
        The SAME single YOLO model processes all frames.
        """
    
        try:
            array = np.frombuffer(
                jpeg_bytes,
                dtype=np.uint8
            )
    
            frame = cv2.imdecode(
                array,
                cv2.IMREAD_COLOR
            )
    
            if frame is None:
                return False
    
            frame = self._resize_frame(frame)
    
            with self.lock:
    
                # Keep only the newest frame.
                # This prevents latency buildup.
                self.phone_frame = frame
                self.phone_frame_seq += 1
    
                # Initialize counter based on actual camera dimensions.
                if self.counter is None:
    
                    fh, fw = frame.shape[:2]
    
                    self.counter = BidirectionalLineCounter(
                        self.config,
                        fw,
                        fh,
                    )
    
                    self.processed_frame = frame.copy()
    
            self.phone_connected = True
    
            return True
    
        except Exception as exc:
    
            logger.warning(
                "Failed to decode browser camera frame: %s",
                exc,
            )
    
            return False

    def reset_counts(self):
        """UI-safe count reset."""
        if self.counter is not None:
            self.counter.reset()

    def update_boundary(self, position) -> bool:
        """
        Accept boundary as either a ratio (0.5) or percentage (50).
        The frontend and backend therefore always use the same line.
        """
        try:
            ratio = float(position)
            if ratio > 1.0:
                ratio /= 100.0

            ratio = max(0.05, min(0.95, ratio))
            self.config.line_position_ratio = ratio

            if self.counter is not None:
                self.counter.set_line_ratio(ratio)

            logger.info(
                "Boundary updated to %.1f%%",
                ratio * 100.0,
            )
            return True

        except (TypeError, ValueError) as exc:
            logger.warning("Invalid boundary value %r: %s", position, exc)
            return False

    def start_capture(self, source: str = "0"):
        self._open_capture(source)

    def stop_capture(self):
        self.is_running = False
        self.phone_connected = False
        self.source_mode = "capture"

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

        self.cap = None

    def stop(self):
        self.stop_capture()

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]

        if self.config.resize_width > 0 and w > self.config.resize_width:
            scale = self.config.resize_width / float(w)
            return cv2.resize(
                frame,
                (
                    self.config.resize_width,
                    max(1, int(h * scale)),
                ),
                interpolation=cv2.INTER_AREA,
            )

        return frame

    async def process_frames(self):
        """
        One processing loop only. The model is never duplicated.
        """
        if not self.model_loaded:
            self.load_model()

        last_process = 0.0

        while self.is_running:
            # -------------------------------------------------------------
            # Get next frame from either OpenCV capture or phone WebSocket.
            # -------------------------------------------------------------
            if self.source_mode in ("phone", "browser_webcam"):
                with self.lock:
                    seq = self.phone_frame_seq
                    frame = (
                        self.phone_frame.copy()
                        if self.phone_frame is not None
                        else None
                    )

                if frame is None or seq == self.phone_processed_seq:
                    await asyncio.sleep(0.005)
                    continue

                # Do not mark this sequence as processed yet.
                # The sender must only see it after YOLO + counter finish.

            else:
                if self.cap is None or not self.cap.isOpened():
                    break

                ok, frame = self.cap.read()

                if not ok or frame is None:
                    logger.warning("Frame read failed. Stopping stream.")
                    break

                frame = self._resize_frame(frame)

            now = time.time()
            min_interval = 1.0 / max(
                1,
                self.config.max_processing_fps,
            )

            if now - last_process < min_interval:
                await asyncio.sleep(0.001)
                continue

            last_process = now

            try:
                # SINGLE MODEL + PERSON CLASS ONLY
                results = self.model.track(
                    frame,
                    persist=True,
                    tracker=self.config.tracker,
                    conf=self.config.conf_threshold,
                    iou=self.config.iou_threshold,
                    classes=[0],
                    verbose=False,
                )

                if self.counter is not None:
                    annotated = self.counter.update(results, frame)
                else:
                    annotated = frame

                with self.lock:
                    self.processed_frame = annotated

                # Mark browser/phone frame complete only AFTER inference.
                if self.source_mode in ("phone", "browser_webcam"):
                    self.phone_processed_seq = seq

            except Exception as exc:
                logger.exception("Frame processing error: %s", exc)
                with self.lock:
                    self.processed_frame = frame

                if self.source_mode in ("phone", "browser_webcam"):
                    self.phone_processed_seq = seq

            await asyncio.sleep(0)

        self.is_running = False
        logger.info("Processing loop stopped")

    def get_frame_jpeg(self) -> Optional[bytes]:
        with self.lock:
            if self.processed_frame is None:
                return None
            frame = self.processed_frame.copy()

        ok, buffer = cv2.imencode(
            ".jpg",
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                int(self.config.jpeg_quality),
            ],
        )

        return buffer.tobytes() if ok else None

    def get_stats(self) -> dict:
        if self.counter is None:
            return {
                "total_count": 0,
                "in_count": 0,
                "out_count": 0,
                "active_tracks": 0,
                "fps": 0,
                "events": [],
                "line_position": round(
                    self.config.line_position_ratio * 100
                ),
                "model": "YOLOv8n-Seg",
                "person_only": True,
            }

        return self.counter.get_stats()


def os_name_is_windows() -> bool:
    return os.name == "nt"


def running_in_cloud_container() -> bool:
    """Best-effort helper for diagnostics only."""
    return bool(
        os.environ.get("RENDER")
        or os.environ.get("RAILWAY_ENVIRONMENT")
        or os.environ.get("KUBERNETES_SERVICE_HOST")
        or platform.system().lower() == "linux" and not any(Path("/dev").glob("video*"))
    )


# =============================================================================
# FASTAPI APP
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

STATIC_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="Vision Counter",
    version="3.0.0",
)

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

config = CounterConfig()
processor: Optional[VideoProcessor] = None


@app.on_event("startup")
async def startup():
    global processor
    processor = VideoProcessor(config)

    # Load the one model once during startup.
    try:
        processor.load_model()
    except Exception as exc:
        processor.model_error = str(exc)
        logger.warning("Model startup error: %s", exc)


@app.on_event("shutdown")
async def shutdown():
    global processor
    if processor is not None:
        processor.stop()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


def get_lan_ip() -> str:
    """Best-effort LAN IP for the QR code when running on a local network."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        return ip
    except Exception:
        return "127.0.0.1"


@app.get("/api/phone-url")
async def phone_url(request: Request):
    """
    Build the phone URL from the URL the user actually opened.

    This is deployment-safe: Render/Railway/VPS/custom-domain/reverse-proxy
    deployments automatically use their public HTTPS host. No static URL,
    localhost URL, ngrok URL, or hard-coded IP is required.
    """
    forwarded_proto = request.headers.get("x-forwarded-proto", "")
    forwarded_host = request.headers.get("x-forwarded-host", "")
    host = forwarded_host or request.headers.get("host", "")
    scheme = forwarded_proto.split(",")[0].strip() or request.url.scheme

    # Production: always use the exact public domain that opened the dashboard.
    # This supports custom domains, Render/Railway, reverse proxies, etc.
    if host:
        host_name = host.split(":")[0].lower()

        # localhost/127.0.0.1 only works on the computer itself. A scanned QR
        # must point to the computer's LAN address instead, because on a phone
        # "localhost" means the phone, not the computer.
        if host_name not in {"localhost", "127.0.0.1", "::1"}:
            return {"url": f"{scheme}://{host}/phone"}

    # Local development fallback. For production deployments this branch is
    # never used; the public HTTPS domain above is used automatically.
    return {"url": f"http://{get_lan_ip()}:8000/phone"}


@app.get("/phone", response_class=HTMLResponse)
async def phone_camera_page():
    """Standalone mobile page opened by the QR code."""
    phone_file = TEMPLATES_DIR / "phone.html"
    if not phone_file.exists():
        return HTMLResponse(
            "<h2>phone.html is missing from templates.</h2>",
            status_code=500,
        )
    return HTMLResponse(phone_file.read_text(encoding="utf-8"))


@app.get("/api/phone-status")
async def phone_status():
    return {
        "connected": bool(
            processor
            and processor.phone_connected
            and processor.source_mode == "phone"
        ),
        "running": bool(processor and processor.is_running),
    }


@app.websocket("/phone-stream")
async def phone_stream_endpoint(websocket: WebSocket):
    """
    Full-duplex phone camera streaming.

    FLOW:

        PHONE CAMERA
             ↓
        JPEG FRAME
             ↓
        /phone-stream
             ↓
        Existing VideoProcessor
             ↓
        Existing SINGLE YOLO model
             ↓
        Person segmentation
             ↓
        ByteTrack
             ↓
        IN / OUT counter
             ↓
        Processed frame + stats
             ↓
        PHONE UI
    """

    await websocket.accept()

    # Same endpoint supports both phone and desktop browser cameras.
    # /phone-stream?mode=phone          -> mobile camera
    # /phone-stream?mode=browser_webcam -> desktop/laptop webcam
    requested_mode = websocket.query_params.get("mode", "phone").strip().lower()
    camera_mode = "browser_webcam" if requested_mode in ("browser_webcam", "webcam", "desktop") else "phone"

    global processor

    sender_task = None

    async def send_results_to_phone():
        """
        Sends the latest YOLO processed frame and statistics
        back to the phone.

        No second YOLO model is created.
        """

        last_sent_frame_seq = -1

        try:
            while True:

                if (
                    processor is not None
                    and processor.is_running
                    and processor.source_mode == camera_mode
                    and processor.phone_connected
                ):

                    # Send only when a new phone frame has been processed.
                    current_seq = processor.phone_processed_seq

                    if (
                        current_seq >= 0
                        and current_seq != last_sent_frame_seq
                    ):

                        jpeg = processor.get_frame_jpeg()

                        if jpeg:

                            stats = processor.get_stats()

                            await websocket.send_json({
                                "type": "frame",

                                "image": base64.b64encode(
                                    jpeg
                                ).decode("utf-8"),

                                "stats": stats,

                                "events": stats.get(
                                    "events",
                                    []
                                ),
                            })

                            last_sent_frame_seq = current_seq

                # Prevent excessive WebSocket messages.
                await asyncio.sleep(0.03)

        except asyncio.CancelledError:
            pass

        except WebSocketDisconnect:
            pass

        except Exception as exc:
            logger.debug(
                "Phone result sender stopped: %s",
                exc
            )


    try:

        if processor is None:

            await websocket.close(
                code=1011
            )

            return


        # ---------------------------------------------------------
        # LOAD EXISTING SINGLE MODEL ONLY
        # ---------------------------------------------------------

        if not processor.model_loaded:

            processor.load_model()


        # ---------------------------------------------------------
        # STOP PREVIOUS SOURCE
        # ---------------------------------------------------------

        processor.stop()


        # Stop old processing task safely.
        if (
            processor.processing_task is not None
            and not processor.processing_task.done()
        ):

            processor.processing_task.cancel()

            try:

                await processor.processing_task

            except asyncio.CancelledError:
                pass

            except Exception:
                pass


        # ---------------------------------------------------------
        # START PHONE CAPTURE
        # ---------------------------------------------------------

        processor.start_browser_capture(camera_mode)


        # Existing processing pipeline.
        processor.processing_task = asyncio.create_task(
            processor.process_frames()
        )


        # Start result sender.
        sender_task = asyncio.create_task(
            send_results_to_phone()
        )


        # Confirm connection.
        await websocket.send_json({

            "type": "status",

            "status": "connected",

            "message":
                f"{'Desktop webcam' if camera_mode == 'browser_webcam' else 'Phone camera'} connected to Vision Counter"

        })


        logger.info(
            "Phone camera WebSocket connected"
        )


        # =========================================================
        # RECEIVE PHONE DATA
        # =========================================================

        while True:

            message = await websocket.receive()


            # -----------------------------------------------------
            # PHONE JPEG FRAME
            # -----------------------------------------------------

            if message.get("bytes") is not None:

                jpeg = message["bytes"]

                if processor is not None:

                    processor.submit_phone_jpeg(
                        jpeg
                    )


            # -----------------------------------------------------
            # CONTROL MESSAGE
            # -----------------------------------------------------

            elif message.get("text") is not None:

                try:

                    data = json.loads(
                        message["text"]
                    )

                    action = data.get(
                        "action"
                    )


                    # STOP
                    if action == "stop":

                        logger.info(
                            "Phone requested stream stop"
                        )

                        break


                    # LIVE CONFIGURATION
                    elif action == "config":

                        cfg = data.get(
                            "config",
                            {}
                        )


                        # Boundary position
                        if "line_position" in cfg:

                            ratio = float(
                                cfg["line_position"]
                            )

                            # Support both 0-1 and 0-100.
                            if ratio > 1:
                                ratio /= 100.0

                            ratio = max(
                                0.05,
                                min(
                                    0.95,
                                    ratio
                                )
                            )

                            config.line_position_ratio = ratio

                            if (
                                processor.counter is not None
                            ):

                                processor.counter.set_line_ratio(
                                    ratio
                                )


                        # Confidence
                        if "conf_threshold" in cfg:

                            config.conf_threshold = max(
                                0.10,
                                min(
                                    0.95,
                                    float(
                                        cfg[
                                            "conf_threshold"
                                        ]
                                    )
                                )
                            )


                        # Hysteresis / dead zone
                        if "hysteresis" in cfg:

                            config.hysteresis_pixels = max(
                                10,
                                min(
                                    150,
                                    int(
                                        cfg[
                                            "hysteresis"
                                        ]
                                    )
                                )
                            )


                        # Masks
                        if "show_masks" in cfg:

                            config.show_masks = bool(
                                cfg[
                                    "show_masks"
                                ]
                            )


                        # Track IDs
                        if "show_track_ids" in cfg:

                            config.show_track_ids = bool(
                                cfg[
                                    "show_track_ids"
                                ]
                            )


                        await websocket.send_json({

                            "type":
                                "config_updated",

                            "stats":
                                processor.get_stats()

                        })


                    # RESET COUNTS
                    elif action == "reset":

                        if (
                            processor.counter is not None
                        ):

                            processor.counter.total_count = 0
                            processor.counter.in_count = 0
                            processor.counter.out_count = 0

                            processor.counter.events.clear()


                        await websocket.send_json({

                            "type": "status",

                            "status": "reset",

                            "message":
                                "Counts reset"

                        })


                except json.JSONDecodeError:

                    logger.warning(
                        "Invalid JSON received from phone"
                    )


    except WebSocketDisconnect:

        logger.info(
            "Phone camera disconnected"
        )


    except asyncio.CancelledError:

        logger.info(
            "Phone stream cancelled"
        )


    except Exception as exc:

        logger.exception(
            "Phone stream error: %s",
            exc
        )

        try:

            await websocket.send_json({

                "type": "error",

                "message": str(exc)

            })

        except Exception:
            pass


    finally:

        # ---------------------------------------------------------
        # STOP RESULT SENDER
        # ---------------------------------------------------------

        if sender_task is not None:

            sender_task.cancel()

            try:

                await sender_task

            except asyncio.CancelledError:
                pass

            except Exception:
                pass


        # ---------------------------------------------------------
        # STOP PHONE PROCESSING
        # ---------------------------------------------------------
        if (
            processor is not None
            and processor.source_mode in ("phone", "browser_webcam")
        ):
            processor.phone_connected = False
            processor.is_running = False


        logger.info(
            "Phone camera session ended"
        )
        
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "running": processor.is_running if processor else False,
        "model_loaded": processor.model_loaded if processor else False,
        "model": config.model_path,
        "person_only": True,
        "source": processor.last_source if processor else None,
        "error": processor.model_error if processor else None,
        "phone_connected": processor.phone_connected if processor else False,
        "server_has_local_camera": any(Path("/dev").glob("video*")) if not os_name_is_windows() else True,
    }


@app.get("/api/sources")
async def sources():
    """
    Source types for the UI selector.

    The UI can show:
    - Local Webcam
    - Phone / IP Camera
    - CCTV / RTSP
    - Video File
    """
    return {
        "sources": [
            {
                "id": "webcam",
                "label": "Local Webcam",
                "placeholder": "0",
                "example": "0",
            },
            {
                "id": "phone",
                "label": "Phone / IP Camera",
                "placeholder": "http://192.168.x.x:8080/video",
                "example": "http://192.168.1.10:8080/video",
            },
            {
                "id": "cctv",
                "label": "CCTV / RTSP",
                "placeholder": "rtsp://username:password@ip:554/stream",
                "example": "rtsp://192.168.1.20:554/stream",
            },
            {
                "id": "video",
                "label": "Video File",
                "placeholder": "C:/path/to/video.mp4",
                "example": "sample.mp4",
            },
        ]
    }


# =============================================================================
# WEBSOCKET
# =============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connected")

    global processor

    sender_task = None

    async def send_results_to_browser():
        """
        Dedicated sender for processed browser webcam frames.

        Frame receiving and YOLO inference are asynchronous, so this task
        continuously checks for newly processed frames and sends them back
        without blocking incoming camera frames.
        """
        last_sent_frame_seq = -1

        try:
            while True:

                if (
                    processor is not None
                    and processor.is_running
                    and processor.source_mode == "browser_webcam"
                    and processor.phone_connected
                ):
                    current_seq = processor.phone_processed_seq

                    if (
                        current_seq >= 0
                        and current_seq != last_sent_frame_seq
                        and processor.processed_frame is not None
                    ):
                        jpeg = processor.get_frame_jpeg()

                        if jpeg:
                            stats = processor.get_stats()

                            await websocket.send_json({
                                "type": "frame",
                                "image": base64.b64encode(
                                    jpeg
                                ).decode("utf-8"),
                                "stats": stats,
                                "events": stats.get("events", [])
                            })

                            last_sent_frame_seq = current_seq

                # Small delay prevents unnecessary CPU usage
                await asyncio.sleep(0.03)

        except asyncio.CancelledError:
            pass

        except WebSocketDisconnect:
            pass

        except Exception as exc:
            logger.debug(
                "Browser result sender stopped: %s",
                exc
            )

    try:
        while True:

            # IMPORTANT:
            # receive() supports BOTH text messages and binary JPEG frames.
            # Do NOT use receive_text() here.
            message = await websocket.receive()

            # ----------------------------------------------------------
            # WEBSOCKET DISCONNECT
            # ----------------------------------------------------------
            if message.get("type") == "websocket.disconnect":
                break

            # ----------------------------------------------------------
            # TEXT / JSON CONTROL MESSAGE
            # ----------------------------------------------------------
            if message.get("text") is not None:

                try:
                    data = json.loads(message["text"])
                except json.JSONDecodeError:
                    logger.warning("Invalid WebSocket JSON received")
                    continue

                action = data.get("action")

                # ------------------------------------------------------
                # START SOURCE
                # ------------------------------------------------------
                if action == "start":
                
                    source = data.get("source")
                
                    if source is None:
                        await websocket.send_json({
                            "type": "error",
                            "message": "No video source provided."
                        })
                        continue
                
                    # ======================================================
                    # CLEAN UP PREVIOUS SOURCE AND TASKS
                    # ======================================================
                
                    # Stop previous browser result sender
                    if sender_task is not None and not sender_task.done():
                
                        sender_task.cancel()
                
                        try:
                            await sender_task
                        except asyncio.CancelledError:
                            pass
                        except Exception:
                            pass
                
                        sender_task = None
                
                
                    # Stop previous capture and processing task
                    if processor is not None:
                
                        processor.stop_capture()
                
                        if (
                            processor.processing_task is not None
                            and not processor.processing_task.done()
                        ):
                
                            processor.processing_task.cancel()
                
                            try:
                                await processor.processing_task
                            except asyncio.CancelledError:
                                pass
                            except Exception:
                                pass
                
                            processor.processing_task = None
                
                
                    # ==================================================
                    # BROWSER WEBCAM
                    # ==================================================
                
                    if source in (
                        "__BROWSER_WEBCAM__",
                        "__BROWSER_CAMERA__",
                        "__WEB_BROWSER_CAMERA__",
                    ):
                
                        processor.start_webcam_capture()
                
                        if (
                            processor.processing_task is None
                            or processor.processing_task.done()
                        ):
                            processor.processing_task = asyncio.create_task(
                                processor.process_frames()
                            )
                
                        if (
                            sender_task is None
                            or sender_task.done()
                        ):
                            sender_task = asyncio.create_task(
                                send_results_to_browser()
                            )
                
                        await websocket.send_json({
                            "type": "status",
                            "message": "Browser webcam started",
                            "source": "browser_webcam"
                        })
                
                        logger.info("Browser webcam source prepared")
                
                    # ==================================================
                    # NORMAL SOURCES
                    # ==================================================
                    else:
                
                        processor.start_capture(source)
                
                        if (
                            processor.processing_task is None
                            or processor.processing_task.done()
                        ):
                            processor.processing_task = asyncio.create_task(
                                processor.process_frames()
                            )
                
                        await websocket.send_json({
                            "type": "status",
                            "message": "Video source started"
                        })

                # ------------------------------------------------------
                # STOP SOURCE
                # ------------------------------------------------------
                elif action == "stop":

                    if processor is not None:
                        processor.stop_capture()

                    await websocket.send_json({
                        "type": "status",
                        "message": "Video source stopped"
                    })

                # ------------------------------------------------------
                # RESET COUNTS
                # ------------------------------------------------------
                elif action == "reset":

                    if processor is not None:
                        processor.reset_counts()

                    await websocket.send_json({
                        "type": "stats",
                        "stats": processor.get_stats()
                        if processor is not None
                        else {}
                    })

                # ------------------------------------------------------
                # UPDATE BOUNDARY
                # ------------------------------------------------------
                elif action == "update_boundary":

                    if processor is not None:
                        boundary = data.get("boundary")
                        if boundary is None:
                            boundary = data.get("line_position")

                        if boundary is not None:
                            processor.update_boundary(boundary)

                        await websocket.send_json({
                            "type": "stats",
                            "stats": processor.get_stats()
                        })

                # ------------------------------------------------------
                # CONFIG UPDATE (supports updated frontend)
                # ------------------------------------------------------
                elif action == "config":

                    if processor is not None:
                        cfg = data.get("config", data)

                        boundary = cfg.get(
                            "line_position",
                            cfg.get("boundary")
                        )
                        if boundary is not None:
                            processor.update_boundary(boundary)

                        if "conf_threshold" in cfg:
                            try:
                                processor.config.conf_threshold = max(
                                    0.05,
                                    min(0.95, float(cfg["conf_threshold"]))
                                )
                            except (TypeError, ValueError):
                                pass

                        if "hysteresis" in cfg:
                            try:
                                processor.config.hysteresis_pixels = max(
                                    10,
                                    min(150, int(cfg["hysteresis"]))
                                )
                            except (TypeError, ValueError):
                                pass

                        if "show_masks" in cfg:
                            processor.config.show_masks = bool(cfg["show_masks"])

                        if "show_track_ids" in cfg:
                            processor.config.show_track_ids = bool(
                                cfg["show_track_ids"]
                            )

                        await websocket.send_json({
                            "type": "config_updated",
                            "stats": processor.get_stats()
                        })

                # ------------------------------------------------------
                # GET CURRENT STATS
                # ------------------------------------------------------
                elif action == "get_stats":

                    if processor is not None:
                        await websocket.send_json({
                            "type": "stats",
                            "stats": processor.get_stats()
                        })

                continue

            # ----------------------------------------------------------
            # BINARY JPEG FRAME FROM BROWSER WEBCAM
            # ----------------------------------------------------------
            if message.get("bytes") is not None:

                frame_bytes = message["bytes"]

                if not frame_bytes:
                    continue

                if processor is None:
                    continue

                # Only accept browser frames when browser webcam
                # is the active source.
                if processor.source_mode != "browser_webcam":
                    continue

                # Submit newest frame.
                # process_frames() handles YOLO asynchronously.
                processor.submit_phone_jpeg(frame_bytes)

                # Do NOT wait for YOLO here.
                # send_results_to_browser() sends completed results.
                continue

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")

    except Exception as exc:
        logger.exception(
            "WebSocket error: %s",
            exc
        )

        try:
            await websocket.send_json({
                "type": "error",
                "message": str(exc)
            })
        except Exception:
            pass

    finally:

        # ----------------------------------------------------------
        # STOP DEDICATED BROWSER RESULT SENDER
        # ----------------------------------------------------------
        if sender_task is not None:
            sender_task.cancel()

            try:
                await sender_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

        # ----------------------------------------------------------
        # CLEAN UP BROWSER CAMERA
        # ----------------------------------------------------------
        if (
            processor is not None
            and processor.source_mode == "browser_webcam"
        ):
            try:
                processor.stop_capture()
            except Exception:
                pass

        logger.info("WebSocket connection closed")
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
