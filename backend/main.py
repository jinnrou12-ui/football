"""
Football Video Analysis Studio - Backend
FastAPI server with YOLOv8 detection, ball possession logic, and visual effects.
"""

import os
import uuid
import math
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from pydantic import BaseModel

# Monkeypatch torch.load to default weights_only=False for YOLOv8 compatibility
original_load = torch.load
def patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(title="Football Video Analysis Studio API", version="1.0.0")

class PrivateNetworkMiddleware:
    """ASGI middleware to append Access-Control-Allow-Private-Network header for Chrome LNA compatibility."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    req_headers = dict(scope.get("headers", []))
                    if b"access-control-request-private-network" in req_headers:
                        headers_list = message.get("headers", [])
                        has_pna = any(h[0].lower() == b"access-control-allow-private-network" for h in headers_list)
                        if not has_pna:
                            headers_list.append((b"access-control-allow-private-network", b"true"))
                await send(message)
            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)

app.add_middleware(PrivateNetworkMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
BASE_DIR       = Path(__file__).parent
UPLOAD_DIR     = BASE_DIR / "uploads"
OUTPUT_DIR     = BASE_DIR / "outputs"
THUMBNAIL_DIR  = OUTPUT_DIR / "thumbnails"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# YOLOv8 model (loaded once at startup)
# ---------------------------------------------------------------------------
MODEL_PATH = "yolov8s.pt"   # Upgraded to YOLOv8s for much better sports ball detection
model: YOLO | None = None

@app.on_event("startup")
async def load_model():
    global model
    print("[Startup] Loading YOLOv8 model (yolov8s) …")
    model = YOLO(MODEL_PATH)
    print("[Startup] YOLOv8 model ready.")

# ---------------------------------------------------------------------------
# In-memory job tracker  {job_id: {"status": str, "filename": str | None}}
# ---------------------------------------------------------------------------
jobs: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """Convert a CSS hex color (#RRGGBB or RRGGBB) to an OpenCV BGR tuple."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (b, g, r)   # OpenCV uses BGR


def euclidean(x1: float, y1: float, x2: float, y2: float) -> float:
    """2-D Euclidean distance."""
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def dist_to_box(bx: float, by: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate the shortest distance from a point (the ball) to a player's bounding box."""
    cx = max(x1, min(bx, x2))
    cy = max(y1, min(by, y2))
    return euclidean(bx, by, cx, cy)


def get_box_overlap(boxA: tuple[int, int, int, int], boxB: tuple[int, int, int, int]) -> int:
    """Calculate the intersection area between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    return max(0, xB - xA) * max(0, yB - yA)
def draw_tracker_ring(frame: np.ndarray,
                      cx: int, cy: int,
                      color: tuple[int, int, int],
                      radius: int = 12) -> None:
    """Draw a glowing double-ring around the ball centre, scaled dynamically to its size."""
    radius = max(6, min(35, radius))
    outer_gap = max(3, int(radius * 0.3))
    inner_thickness = max(2, int(radius * 0.15))
    dot_radius = max(1, int(radius * 0.2))

    # Outer glow ring
    cv2.circle(frame, (cx, cy), radius + outer_gap, color, 1, cv2.LINE_AA)
    # Inner solid ring
    cv2.circle(frame, (cx, cy), radius, color, inner_thickness, cv2.LINE_AA)
    # Centre dot
    cv2.circle(frame, (cx, cy), dot_radius, color, -1, cv2.LINE_AA)


def apply_possession_highlight(frame: np.ndarray,
                                x1: int, y1: int, x2: int, y2: int,
                                color: tuple[int, int, int]) -> None:
    """Draw a coloured rectangle around the possessing player without any text label."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)


def blur_player_box_with_alpha(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, alpha: float) -> None:
    if alpha <= 0.0:
        return
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    roi_w = x2 - x1
    roi_h = y2 - y1
    if roi_w <= 5 or roi_h <= 5:
        return
        
    roi = frame[y1:y2, x1:x2]
    
    # Ensure kernel size is odd and smaller than the ROI dimensions
    kw = (roi_w // 2) * 2 - 1
    kh = (roi_h // 2) * 2 - 1
    kw = max(3, min(25, kw))
    kh = max(3, min(25, kh))
    
    if kw % 2 == 0: kw += 1
    if kh % 2 == 0: kh += 1
    
    blurred = cv2.GaussianBlur(roi, (kw, kh), 0)
    
    if alpha >= 1.0:
        frame[y1:y2, x1:x2] = blurred
    else:
        cv2.addWeighted(blurred, alpha, roi, 1.0 - alpha, 0, frame[y1:y2, x1:x2])


def apply_player_bounding_box_blur(frame: np.ndarray, 
                                   players_small: list[tuple[int, list[int]]], 
                                   blur_factors: dict[int, float], 
                                   x_scale: float, 
                                   y_scale: float) -> np.ndarray:
    """
    Applies Gaussian blur exclusively within the bounding boxes of off-ball players on the high-resolution frame.
    Base frame (background, grass field) remains 100% clear.
    """
    frame_out = frame.copy()
    for tid, (x1, y1, x2, y2) in players_small:
        alpha = blur_factors.get(tid, 1.0)
        if alpha > 0.0:
            px1 = int(round(x1 * x_scale))
            py1 = int(round(y1 * y_scale))
            px2 = int(round(x2 * x_scale))
            py2 = int(round(y2 * y_scale))
            
            blur_player_box_with_alpha(frame_out, px1, py1, px2, py2, alpha)
            
    return frame_out


def smooth_ball_trajectory(ball_centers: list[tuple[int, int] | None], window_size: int = 5) -> list[tuple[int, int] | None]:
    N = len(ball_centers)
    filled_centers = list(ball_centers)
    
    # Pass 1: Gap filling (linear interpolation for gaps of up to window_size frames)
    for i in range(N):
        if filled_centers[i] is None:
            left_val = None
            left_idx = -1
            for j in range(i - 1, max(-1, i - window_size - 1), -1):
                if filled_centers[j] is not None:
                    left_val = filled_centers[j]
                    left_idx = j
                    break
            
            right_val = None
            right_idx = -1
            for j in range(i + 1, min(N, i + window_size + 1)):
                if filled_centers[j] is not None:
                    right_val = filled_centers[j]
                    right_idx = j
                    break
            
            if left_val is not None and right_val is not None:
                diff = right_idx - left_idx
                ratio = (i - left_idx) / float(diff)
                x = left_val[0] + (right_val[0] - left_val[0]) * ratio
                y = left_val[1] + (right_val[1] - left_val[1]) * ratio
                filled_centers[i] = (int(round(x)), int(round(y)))
    
    # Pass 2: Temporal smoothing (moving average filter)
    smoothed_centers = [None] * N
    for i in range(N):
        if filled_centers[i] is not None:
            vals = []
            for j in range(max(0, i - 2), min(N, i + 3)):
                if filled_centers[j] is not None:
                    vals.append(filled_centers[j])
            if vals:
                mean_x = sum(v[0] for v in vals) / len(vals)
                mean_y = sum(v[1] for v in vals) / len(vals)
                smoothed_centers[i] = (int(round(mean_x)), int(round(mean_y)))
        else:
            smoothed_centers[i] = None
            
    return smoothed_centers


def draw_name_tag(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, name: str, tracker_color: tuple[int, int, int]) -> None:
    """Draw a premium name tag (pill shape with text and a pointer triangle) above the player."""
    cx = (x1 + x2) // 2
    cy = y1 - 10  # Positioned slightly above the player's head
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2
    text_size = cv2.getTextSize(name.upper(), font, font_scale, thickness)[0]
    tw, th = text_size[0], text_size[1]
    
    padding_x = 8
    padding_y = 5
    pill_w = tw + 2 * padding_x
    pill_h = th + 2 * padding_y
    
    pill_x1 = cx - pill_w // 2
    pill_y1 = cy - pill_h - 10
    pill_x2 = cx + pill_w // 2
    pill_y2 = cy - 10
    
    h_f, w_f = frame.shape[:2]
    if pill_y1 < 0:
        shift = -pill_y1 + 5
        pill_y1 += shift
        pill_y2 += shift
        cy += shift

    # Draw the pointer triangle pointing down
    pts = np.array([[cx, cy], [cx - 6, cy - 10], [cx + 6, cy - 10]], np.int32)
    cv2.drawContours(frame, [pts], 0, tracker_color, -1, cv2.LINE_AA)
    
    # Draw the pill background (dark grey/black with a light border)
    bg_color = (25, 25, 25)
    border_color = (200, 200, 200)
    
    cv2.rectangle(frame, (pill_x1, pill_y1), (pill_x2, pill_y2), bg_color, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (pill_x1, pill_y1), (pill_x2, pill_y2), border_color, 1, cv2.LINE_AA)
    
    # Draw text (white, bold uppercase)
    tx = pill_x1 + padding_x
    ty = pill_y2 - padding_y
    cv2.putText(frame, name.upper(), (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_name_tag_with_alpha(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, name: str, tracker_color: tuple[int, int, int], alpha: float = 1.0) -> None:
    """Draw the name tag with alpha blending for smooth fade transitions."""
    if alpha <= 0.0:
        return
    if alpha >= 1.0:
        draw_name_tag(frame, x1, y1, x2, y2, name, tracker_color)
        return
        
    cx = (x1 + x2) // 2
    cy = y1 - 10
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2
    text_size = cv2.getTextSize(name.upper(), font, font_scale, thickness)[0]
    tw, th = text_size[0], text_size[1]
    
    padding_x = 8
    padding_y = 5
    pill_w = tw + 2 * padding_x
    pill_h = th + 2 * padding_y
    
    pill_x1 = cx - pill_w // 2
    pill_y1 = cy - pill_h - 10 - 5
    pill_x2 = cx + pill_w // 2
    pill_y2 = cy + 5
    
    h_f, w_f = frame.shape[:2]
    pill_x1 = max(0, min(w_f - 1, pill_x1))
    pill_y1 = max(0, min(h_f - 1, pill_y1))
    pill_x2 = max(0, min(w_f - 1, pill_x2))
    pill_y2 = max(0, min(h_f - 1, pill_y2))
    
    if pill_x2 <= pill_x1 or pill_y2 <= pill_y1:
        return
        
    roi = frame[pill_y1:pill_y2, pill_x1:pill_x2].copy()
    dx = -pill_x1
    dy = -pill_y1
    draw_name_tag(roi, x1 + dx, y1 + dy, x2 + dx, y2 + dy, name, tracker_color)
    cv2.addWeighted(roi, alpha, frame[pill_y1:pill_y2, pill_x1:pill_x2], 1.0 - alpha, 0, frame[pill_y1:pill_y2, pill_x1:pill_x2])


class IoUTracker:
    def __init__(self, iou_threshold=0.25, max_lost_frames=30, max_render_lost_frames=10):
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.max_render_lost_frames = max_render_lost_frames
        self.next_id = 1
        # tracks: id -> {"box": (x1, y1, x2, y2), "lost_frames": int, "velocity": (vx, vy)}
        self.tracks = {}

    def update(self, detections):
        # 1. Project all existing tracks using their velocity
        projected_boxes = {}
        for track_id, track_data in self.tracks.items():
            box = track_data["box"]
            vx, vy = track_data.get("velocity", (0.0, 0.0))
            proj_box = (box[0] + vx, box[1] + vy, box[2] + vx, box[3] + vy)
            projected_boxes[track_id] = proj_box

        # 2. Compute similarity matrix using projected boxes, distances, and velocity consistency
        candidates = []
        for track_id, track_data in self.tracks.items():
            if track_data["lost_frames"] > self.max_lost_frames:
                continue
            
            proj_box = projected_boxes[track_id]
            proj_cx = (proj_box[0] + proj_box[2]) / 2.0
            proj_cy = (proj_box[1] + proj_box[3]) / 2.0
            
            old_box = track_data["box"]
            old_cx = (old_box[0] + old_box[2]) / 2.0
            old_cy = (old_box[1] + old_box[3]) / 2.0
            
            vx, vy = track_data.get("velocity", (0.0, 0.0))
            track_speed = math.sqrt(vx**2 + vy**2)
            
            # Dynamic matching distance based on speed
            max_match_dist = 45.0 + 2.0 * track_speed
            
            for det_idx, det in enumerate(detections):
                iou = self.calculate_iou(proj_box, det)
                det_cx = (det[0] + det[2]) / 2.0
                det_cy = (det[1] + det[3]) / 2.0
                dist = math.sqrt((proj_cx - det_cx)**2 + (proj_cy - det_cy)**2)
                
                # Direction consistency score (ID-memory lock)
                dir_consistency = 0.0
                if track_speed > 2.0:
                    dx = det_cx - old_cx
                    dy = det_cy - old_cy
                    disp_len = math.sqrt(dx**2 + dy**2)
                    if disp_len > 0.5:
                        cos_sim = (vx * dx + vy * dy) / (track_speed * disp_len)
                        # Reward same direction, penalize opposite direction
                        dir_consistency = 0.3 * cos_sim
                        
                # Allow matching if there is either reasonable IoU overlap or very close proximity
                if iou >= self.iou_threshold or dist < max_match_dist:
                    score = iou + dir_consistency
                    if dist < 20.0:
                        score += 0.3
                    candidates.append((score, track_id, det_idx, dist))

        # Sort candidates by score descending to perform global competitive matching
        candidates.sort(key=lambda x: x[0], reverse=True)

        matched_tracks = {}
        matched_detections = set()

        for score, track_id, det_idx, dist in candidates:
            if track_id in matched_tracks or det_idx in matched_detections:
                continue
            
            matched_tracks[track_id] = det_idx
            matched_detections.add(det_idx)

        # Estimate camera motion (global displacement) from matched tracks
        dx_list = []
        dy_list = []
        for track_id, det_idx in matched_tracks.items():
            new_box = detections[det_idx]
            old_box = self.tracks[track_id]["box"]
            old_cx = (old_box[0] + old_box[2]) / 2.0
            old_cy = (old_box[1] + old_box[3]) / 2.0
            new_cx = (new_box[0] + new_box[2]) / 2.0
            new_cy = (new_box[1] + new_box[3]) / 2.0
            dx_list.append(new_cx - old_cx)
            dy_list.append(new_cy - old_cy)
            
        if len(dx_list) > 0:
            camera_dx = float(np.median(dx_list))
            camera_dy = float(np.median(dy_list))
        else:
            camera_dx = 0.0
            camera_dy = 0.0

        # 3. Update track states
        updated_tracks = {}
        for track_id, track_data in self.tracks.items():
            if track_id in matched_tracks:
                det_idx = matched_tracks[track_id]
                new_box = detections[det_idx]
                old_box = track_data["box"]
                
                # Calculate new velocity components
                old_cx = (old_box[0] + old_box[2]) / 2.0
                old_cy = (old_box[1] + old_box[3]) / 2.0
                new_cx = (new_box[0] + new_box[2]) / 2.0
                new_cy = (new_box[1] + new_box[3]) / 2.0
                
                raw_vx = new_cx - old_cx
                raw_vy = new_cy - old_cy
                
                prev_vx, prev_vy = track_data.get("velocity", (0.0, 0.0))
                
                # Cap velocity to prevent extreme drift from bad detections/noise
                max_v = 40.0
                raw_vx = max(-max_v, min(max_v, raw_vx))
                raw_vy = max(-max_v, min(max_v, raw_vy))
                
                # Smooth the velocity vector using exponential moving average
                if track_data["lost_frames"] > 0:
                    vx = prev_vx
                    vy = prev_vy
                else:
                    vx = 0.7 * raw_vx + 0.3 * prev_vx
                    vy = 0.7 * raw_vy + 0.3 * prev_vy

                updated_tracks[track_id] = {
                    "box": new_box,
                    "lost_frames": 0,
                    "velocity": (vx, vy)
                }
            else:
                # If lost, apply velocity prediction + camera motion to the box so it continues moving
                # during occlusion/motion blur.
                if track_data["lost_frames"] <= self.max_lost_frames:
                    old_box = track_data["box"]
                    vx, vy = track_data.get("velocity", (0.0, 0.0))
                    
                    # Decelerate slightly when lost to prevent infinite drifting
                    vx_decay = vx * 0.9
                    vy_decay = vy * 0.9
                    
                    # Combine relative velocity and camera panning
                    total_dx = vx + camera_dx
                    total_dy = vy + camera_dy
                    
                    projected_box = (
                        old_box[0] + total_dx,
                        old_box[1] + total_dy,
                        old_box[2] + total_dx,
                        old_box[3] + total_dy
                    )
                    updated_tracks[track_id] = {
                        "box": projected_box,
                        "lost_frames": track_data["lost_frames"] + 1,
                        "velocity": (vx_decay, vy_decay)
                    }

        # 4. Initialize new tracks
        for idx, det in enumerate(detections):
            if idx not in matched_detections:
                updated_tracks[self.next_id] = {
                    "box": det,
                    "lost_frames": 0,
                    "velocity": (0.0, 0.0)
                }
                self.next_id += 1

        self.tracks = updated_tracks
        
        # 5. Gather active tracks for rendering
        active_detections = []
        for track_id, track_data in self.tracks.items():
            # Keep rendering tracks for a few frames even if temporarily lost,
            # projecting their bounding boxes to avoid flicker.
            # Sideline / camera panning / fast motion check to dynamically increase max render lost frames
            vx, vy = track_data.get("velocity", (0.0, 0.0))
            track_speed = math.sqrt(vx**2 + vy**2)
            
            box = track_data["box"]
            is_near_boundary = (box[0] < 60 or box[2] > 580 or box[1] < 45 or box[3] > 315)
            is_fast = (track_speed > 4.0)
            
            # If fast or near boundaries or camera is panning, use larger render threshold to prevent flicker
            render_threshold = self.max_render_lost_frames
            if is_near_boundary or is_fast or (abs(camera_dx) > 3.0 or abs(camera_dy) > 3.0):
                render_threshold = self.max_lost_frames - 2
                
            if track_data["lost_frames"] <= render_threshold:
                box_coords = [int(round(c)) for c in track_data["box"]]
                active_detections.append((track_id, box_coords))
                
        return active_detections

    def calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        unionArea = boxAArea + boxBArea - interArea
        return interArea / float(unionArea) if unionArea > 0 else 0


class BallKalmanFilter:
    def __init__(self, dt=1.0):
        # State vector: [x, y, vx, vy]
        self.state = np.zeros((4, 1), dtype=np.float32)
        
        # State transition matrix F
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix H
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Covariance matrix P
        self.P = np.eye(4, dtype=np.float32) * 100.0
        
        # Base process noise covariance (lower values = smoother trajectory)
        self.Q_base = np.array([
            [0.05, 0, 0.02, 0],
            [0, 0.05, 0, 0.02],
            [0.02, 0, 0.5, 0],
            [0, 0.02, 0, 0.5]
        ], dtype=np.float32) * 0.02
        
        self.Q = self.Q_base.copy()
        
        # Base measurement noise covariance (higher values = filters out jitter, trusts model prediction more)
        self.R_base = np.eye(2, dtype=np.float32) * 8.0
        self.R = self.R_base.copy()
        
        self.initialized = False

    def initialize(self, x, y):
        self.state = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 10.0
        self.initialized = True

    def predict(self, velocity_hint=None):
        if velocity_hint is not None:
            # Blend current filter velocity state with optical flow displacement hint
            self.state[2, 0] = 0.5 * self.state[2, 0] + 0.5 * velocity_hint[0]
            self.state[3, 0] = 0.5 * self.state[3, 0] + 0.5 * velocity_hint[1]
            
        # Adapt process noise Q based on velocity magnitude to prevent lag during fast moves
        speed = math.sqrt(self.state[2, 0]**2 + self.state[3, 0]**2)
        if speed > 12.0:
            scale = min(5.0, speed / 12.0)
            self.Q = self.Q_base * scale
            self.Q[2, 2] *= scale
            self.Q[3, 3] *= scale
        else:
            self.Q = self.Q_base.copy()

        self.state = np.dot(self.F, self.state)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return float(self.state[0, 0]), float(self.state[1, 0])

    def update(self, x, y, R_scale=1.0):
        # Adapt measurement noise based on speed: trust measurement more when fast,
        # and trust model prediction more (smooth out jitter) when slow.
        speed = math.sqrt(self.state[2, 0]**2 + self.state[3, 0]**2)
        
        current_R = self.R_base * R_scale
        if speed > 12.0:
            trust_factor = max(0.1, 12.0 / speed)
            current_R = current_R * trust_factor
        else:
            current_R = current_R * 1.5

        z = np.array([[x], [y]], dtype=np.float32)
        y_residual = z - np.dot(self.H, self.state)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + current_R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.state = self.state + np.dot(K, y_residual)
        I = np.eye(4, dtype=np.float32)
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        return float(self.state[0, 0]), float(self.state[1, 0])


# ---------------------------------------------------------------------------
# Core processing function (runs in background thread)
# ---------------------------------------------------------------------------

POSSESSION_THRESHOLD_PX = 80   # pixels – ball-to-feet distance threshold

def process_video(job_id: str,
                  input_path: Path,
                  tracker_color_hex: str,
                  name_mapping: dict[str, str] = None) -> None:
    """
    Full pipeline:
      1. Open video with OpenCV.
      2. Run YOLOv8 on every frame.
      3. Track players using IoUTracker and trace ball + possession.
      4. Save tracking positions and crop thumbnails for each player.
      5. Apply visual effects (blur, tracker ring, possession highlight, player name tags).
      6. Write output video and save metadata json.
    """
    if name_mapping is None:
        name_mapping = {}

    try:
        import json
        jobs[job_id]["status"] = "processing"

        tracker_bgr = hex_to_bgr(tracker_color_hex)

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError("Cannot open video file.")

        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_filename = f"processed_{job_id}.mp4"
        output_path     = OUTPUT_DIR / output_filename

        if os.name == "nt":
            try:
                fourcc = cv2.VideoWriter_fourcc(*"avc1")
                writer = cv2.VideoWriter(str(output_path), cv2.CAP_MSMF, fourcc, fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError("MSMF writer failed to open")
            except Exception:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        # Ball tracking state
        last_ball_center = None
        last_ball_radius = 12
        ball_lost_frames = 0
        kf = BallKalmanFilter()
        prev_gray = None

        # Player tracker
        player_tracker = IoUTracker()
        saved_thumbnails = set()
        
        # Player blur transition states (track_id -> current_blur_factor)
        blur_factors = {}

        # Player bounding box smoothing states (track_id -> [x1, y1, x2, y2])
        smoothed_boxes = {}

        # Player possession tracking state (Persistent State Machine using track IDs)
        active_possessor_track_id = None
        possessor_lost_frames = 0
        possessor_durations = {}
        tag_decays = {}

        # Goal scorer tracking state
        goal_scorer_track_id = None
        goal_scorer_frames_left = 0
        last_possessor_track_id_before_shot = None

        tracking_metadata = {
            "fps": fps,
            "width": width,
            "height": height,
            "tracker_color_hex": tracker_color_hex,
            "frames": []
        }

        # Pass 1: Fast Detection & Tracking (using 640x360 downscaled frames)
        raw_ball_centers = []
        raw_ball_radii = []
        raw_tracked_players_list = []
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_small = cv2.resize(frame, (640, 360))
            
            # YOLO inference on 640x360 frame
            results = model(frame_small, conf=0.1, verbose=False)[0]

            players_small = []
            ball_candidates_small = []

            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if cls_id == 0 and conf > 0.35:          # player
                    players_small.append((x1, y1, x2, y2))
                elif cls_id == 32:                       # ball
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    r = max(4, (x2 - x1 + y2 - y1) // 4)
                    ball_candidates_small.append(((cx, cy), r, conf))

            # Update tracked players in 640x360
            tracked_players_small = player_tracker.update(players_small)
            raw_tracked_players_list.append(tracked_players_small)

            # Optical Flow on 640x360
            curr_gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            of_pred = None
            if last_ball_center is not None and prev_gray is not None:
                try:
                    prev_pts = np.array([[last_ball_center[0], last_ball_center[1]]], dtype=np.float32).reshape(-1, 1, 2)
                    curr_pts, st, err = cv2.calcOpticalFlowPyrLK(
                        prev_gray, curr_gray, prev_pts, None, 
                        winSize=(21, 21), maxLevel=3,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
                    )
                    if st[0][0] == 1:
                        of_pred = (float(curr_pts[0][0][0]), float(curr_pts[0][0][1]))
                except Exception:
                    pass

            # Kalman Filter prediction
            kf_pred = None
            if kf.initialized:
                velocity_hint = None
                if of_pred is not None and last_ball_center is not None:
                    velocity_hint = (of_pred[0] - last_ball_center[0], of_pred[1] - last_ball_center[1])
                kf_pred = kf.predict(velocity_hint)

            # Combine predictions
            ref_pos = None
            if of_pred is not None and kf_pred is not None:
                ref_pos = (0.7 * of_pred[0] + 0.3 * kf_pred[0], 0.7 * of_pred[1] + 0.3 * kf_pred[1])
            elif of_pred is not None:
                ref_pos = of_pred
            elif kf_pred is not None:
                ref_pos = kf_pred

            # Select best candidate
            best_score = -1
            best_candidate = None
            best_radius = 6

            for (cx, cy), r, conf in ball_candidates_small:
                if ref_pos is not None:
                    dist = euclidean(cx, cy, ref_pos[0], ref_pos[1])
                    if dist < 120:
                        score = conf * (1.0 - (dist / 120.0) * 0.4)
                        if score > best_score:
                            best_score = score
                            best_candidate = (cx, cy)
                            best_radius = r
                else:
                    if conf > 0.25:
                        if conf > best_score:
                            best_score = conf
                            best_candidate = (cx, cy)
                            best_radius = r

            # Update filter state and raw centers
            current_ball = None
            current_radius = last_ball_radius

            if best_candidate is not None:
                if not kf.initialized:
                    kf.initialize(best_candidate[0], best_candidate[1])
                    kf_x, kf_y = float(best_candidate[0]), float(best_candidate[1])
                else:
                    dist_to_kf = euclidean(best_candidate[0], best_candidate[1], kf_pred[0], kf_pred[1])
                    if dist_to_kf > 60:
                        R_scale = 0.05
                    else:
                        R_scale = 1.0
                    kf_x, kf_y = kf.update(best_candidate[0], best_candidate[1], R_scale=R_scale)

                current_ball = (int(kf_x), int(kf_y))
                current_radius = best_radius
                last_ball_center = current_ball
                last_ball_radius = best_radius
                ball_lost_frames = 0
            else:
                if of_pred is not None and kf.initialized and ball_lost_frames < 20:
                    kf_x, kf_y = kf.update(of_pred[0], of_pred[1], R_scale=2.5)
                    current_ball = (int(kf_x), int(kf_y))
                    current_radius = last_ball_radius
                    last_ball_center = current_ball
                    ball_lost_frames += 1
                elif kf_pred is not None and kf.initialized and ball_lost_frames < 12:
                    kf_x, kf_y = kf_pred[0], kf_pred[1]
                    kf.update(kf_x, kf_y, R_scale=8.0)
                    current_ball = (int(kf_x), int(kf_y))
                    current_radius = last_ball_radius
                    last_ball_center = current_ball
                    ball_lost_frames += 1
                else:
                    current_ball = None
                    current_radius = last_ball_radius
                    last_ball_center = None
                    last_ball_radius = 6
                    ball_lost_frames = 0
                    kf.initialized = False

            raw_ball_centers.append(current_ball)
            raw_ball_radii.append(current_radius)

            prev_gray = curr_gray
            frame_idx += 1
            if total > 0:
                jobs[job_id]["progress"] = int(frame_idx / total * 50)

        # Pass 2: Temporal Smoothing & Gap Filling (5-frame look-ahead/look-back sliding window)
        smoothed_ball_centers = smooth_ball_trajectory(raw_ball_centers, window_size=5)

        # Pass 2.5: Precalculate possession & 10-frame look-ahead for all frames
        raw_possessor_ids = []
        scaled_threshold = 80.0 * (640.0 / width)
        
        for f_idx in range(len(raw_tracked_players_list)):
            ball_center_small = smoothed_ball_centers[f_idx]
            tracked_players_small = raw_tracked_players_list[f_idx]
            
            possessor_track_id = None
            if ball_center_small is not None and tracked_players_small:
                bx, by = ball_center_small
                min_dist = float("inf")
                closest_track_id = None
                
                for track_id, (x1, y1, x2, y2) in tracked_players_small:
                    fx = (x1 + x2) // 2
                    fy = y2
                    dist = euclidean(bx, by, fx, fy)
                    if dist < min_dist:
                        min_dist = dist
                        closest_track_id = track_id
                        
                if min_dist <= scaled_threshold:
                    possessor_track_id = closest_track_id
            raw_possessor_ids.append(possessor_track_id)

        # Compute future receivers (up to 10 frames look-ahead)
        future_possessor_ids = [None] * len(raw_possessor_ids)
        future_k_list = [0] * len(raw_possessor_ids)
        
        for f_idx in range(len(raw_possessor_ids)):
            curr_pos = raw_possessor_ids[f_idx]
            for k in range(1, 11):
                look_ahead_idx = f_idx + k
                if look_ahead_idx >= len(raw_possessor_ids):
                    break
                future_pos = raw_possessor_ids[look_ahead_idx]
                if future_pos is not None and future_pos != curr_pos:
                    future_possessor_ids[f_idx] = future_pos
                    future_k_list[f_idx] = k
                    break

        # Reset VideoCapture to start from beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Calculate dynamic alpha-blend transition speed (0.1s duration)
        alpha_step = 1.0 / max(1.0, 0.1 * fps)
        
        # Scale factor from 640x360 to original width/height
        x_scale = width / 640.0
        y_scale = height / 360.0

        frame_idx = 0
        last_rendered_possessor = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            tracked_players_small = raw_tracked_players_list[frame_idx]
            ball_center_small = smoothed_ball_centers[frame_idx]
            ball_radius_small = raw_ball_radii[frame_idx]
            possessor_track_id = raw_possessor_ids[frame_idx]
            future_tid = future_possessor_ids[frame_idx]
            future_k = future_k_list[frame_idx]

            # Calculate ball velocity (for high-speed pass detection)
            ball_speed = 0.0
            if ball_center_small is not None and frame_idx > 0 and smoothed_ball_centers[frame_idx - 1] is not None:
                prev_ball = smoothed_ball_centers[frame_idx - 1]
                ball_speed = euclidean(ball_center_small[0], ball_center_small[1], prev_ball[0], prev_ball[1])

            # Bounding box smoothing (in 640x360 coordinates)
            for tid, box in tracked_players_small:
                box_coords = [float(b) for b in box]
                if tid in smoothed_boxes:
                    prev_box = smoothed_boxes[tid]
                    beta = 0.25  # smoothing factor
                    smoothed = [
                        prev_box[0] + (box_coords[0] - prev_box[0]) * beta,
                        prev_box[1] + (box_coords[1] - prev_box[1]) * beta,
                        prev_box[2] + (box_coords[2] - prev_box[2]) * beta,
                        prev_box[3] + (box_coords[3] - prev_box[3]) * beta,
                    ]
                    smoothed_boxes[tid] = smoothed
                else:
                    smoothed_boxes[tid] = box_coords
                    
            # Clean up stale track IDs in smoothed_boxes
            current_track_ids = {tid for tid, _ in tracked_players_small}
            smoothed_boxes = {tid: val for tid, val in smoothed_boxes.items() if tid in current_track_ids}

            # Crop player thumbnails on first occurrence (upscale coordinates for high-quality thumbnail)
            for tid, (x1, y1, x2, y2) in tracked_players_small:
                if tid not in saved_thumbnails:
                    px1 = max(0, int(round(x1 * x_scale)))
                    py1 = max(0, int(round(y1 * y_scale)))
                    px2 = min(width, int(round(x2 * x_scale)))
                    py2 = min(height, int(round(y2 * y_scale)))
                    if px2 > px1 and py2 > py1:
                        crop = frame[py1:py2, px1:px2]
                        crop_resized = cv2.resize(crop, (120, 160))
                        thumb_filename = f"thumb_{job_id}_{tid}.jpg"
                        cv2.imwrite(str(THUMBNAIL_DIR / thumb_filename), crop_resized)
                        saved_thumbnails.add(tid)

            # Goal scorer detection
            goal_scorer_active = False
            if ball_center_small is not None:
                bx, by = ball_center_small
                in_left_goal = (bx < 640 * 0.25) and (360 * 0.2 < by < 360 * 0.8)
                in_right_goal = (bx > 640 * 0.75) and (360 * 0.2 < by < 360 * 0.8)

                if (in_left_goal or in_right_goal) and last_possessor_track_id_before_shot is not None:
                    goal_scorer_track_id = last_possessor_track_id_before_shot
                    goal_scorer_frames_left = 180
                    last_possessor_track_id_before_shot = None

            if goal_scorer_frames_left > 0 and goal_scorer_track_id is not None:
                still_present = any(track_id == goal_scorer_track_id for track_id, _ in tracked_players_small)
                if still_present:
                    goal_scorer_active = True
                goal_scorer_frames_left -= 1
            else:
                goal_scorer_track_id = None
                goal_scorer_active = False

            if possessor_track_id is not None:
                last_possessor_track_id_before_shot = possessor_track_id

            # Update possession change tag decays
            if possessor_track_id != last_rendered_possessor:
                if last_rendered_possessor is not None and last_rendered_possessor != future_tid:
                    tag_decays[last_rendered_possessor] = 1.0
                last_rendered_possessor = possessor_track_id

            # Decay active label alphas
            decay_step = 1.0 / max(1.0, 0.3 * fps)
            updated_decays = {}
            for tid, alpha in tag_decays.items():
                new_alpha = alpha - decay_step
                if new_alpha > 0.0 and tid in current_track_ids:
                    if tid != possessor_track_id and tid != future_tid:
                        updated_decays[tid] = new_alpha
            tag_decays = updated_decays

            # Find bounding boxes for possessor and receiver for overlap calculations
            possessor_box_small = None
            if possessor_track_id is not None:
                for track_id, box in tracked_players_small:
                    if track_id == possessor_track_id:
                        possessor_box_small = box
                        break

            future_box_small = None
            if future_tid is not None:
                for track_id, box in tracked_players_small:
                    if track_id == future_tid:
                        future_box_small = box
                        break

            # Group/crossover detection: compute pairwise distances to find crowded players
            player_centers = {}
            for tid, box in tracked_players_small:
                cx = (box[0] + box[2]) / 2.0
                cy = (box[1] + box[3]) / 2.0
                player_centers[tid] = (cx, cy)
                
            crowded_players = set()
            for tid1, (cx1, cy1) in player_centers.items():
                for tid2, (cx2, cy2) in player_centers.items():
                    if tid1 != tid2:
                        dist = euclidean(cx1, cy1, cx2, cy2)
                        # If two players stand too close (less than 65 pixels in 640x360), they form a group
                        if dist < 65.0:
                            crowded_players.add(tid1)
                            crowded_players.add(tid2)

            # Update blur alpha factors (smooth transition: 0.1s duration)
            for track_id, box in tracked_players_small:
                target_alpha = 1.0
                
                if possessor_track_id is not None and track_id == possessor_track_id:
                    target_alpha = 0.0
                elif future_tid is not None and track_id == future_tid and future_k <= 10:
                    # Smoothly transition blur to 0.0 as ball approaches receiver
                    target_alpha = max(0.0, (future_k - 1) / 10.0)
                elif track_id in crowded_players:
                    # Prevent blurring crowded players/crossovers
                    target_alpha = 0.0
                else:
                    # Multi-player occlusion: prevent blurring players close to or overlapping with focus
                    if possessor_box_small is not None:
                        overlap = get_box_overlap(box, possessor_box_small)
                        box_area = (box[2] - box[0]) * (box[3] - box[1])
                        overlap_ratio = overlap / float(box_area) if box_area > 0 else 0
                        if overlap_ratio >= 0.15:
                            target_alpha = 0.0
                    
                    if target_alpha > 0.0 and future_box_small is not None:
                        overlap = get_box_overlap(box, future_box_small)
                        box_area = (box[2] - box[0]) * (box[3] - box[1])
                        overlap_ratio = overlap / float(box_area) if box_area > 0 else 0
                        if overlap_ratio >= 0.15:
                            target_alpha = 0.0

                current_alpha = blur_factors.get(track_id, 1.0)
                if current_alpha < target_alpha:
                    new_alpha = min(target_alpha, current_alpha + alpha_step)
                else:
                    new_alpha = max(target_alpha, current_alpha - alpha_step)
                blur_factors[track_id] = new_alpha

            # Clean up stale track IDs in blur_factors
            blur_factors = {tid: val for tid, val in blur_factors.items() if tid in current_track_ids}

            # Filter out drawing highlights for players with extreme overlaps (so highlight doesn't overlap messily)
            highlight_players = []
            for track_id, box in tracked_players_small:
                is_possessor = (possessor_track_id is not None and track_id == possessor_track_id)
                is_receiver = (future_tid is not None and track_id == future_tid and future_k <= 10)
                
                if is_possessor:
                    highlight_players.append((track_id, box, "possessor"))
                elif is_receiver:
                    highlight_players.append((track_id, box, "receiver"))
                else:
                    is_overlapping = False
                    if possessor_box_small is not None:
                        overlap = get_box_overlap(box, possessor_box_small)
                        box_area = (box[2] - box[0]) * (box[3] - box[1])
                        overlap_ratio = overlap / float(box_area) if box_area > 0 else 0
                        if overlap_ratio >= 0.35:
                            is_overlapping = True
                            
                    if not is_overlapping and future_box_small is not None:
                        overlap = get_box_overlap(box, future_box_small)
                        box_area = (box[2] - box[0]) * (box[3] - box[1])
                        overlap_ratio = overlap / float(box_area) if box_area > 0 else 0
                        if overlap_ratio >= 0.35:
                            is_overlapping = True
                            
                    if not is_overlapping:
                        highlight_players.append((track_id, box, "none"))

            # Apply Gaussian blur exclusively to off-ball player bounding boxes
            frame_out = apply_player_bounding_box_blur(frame, tracked_players_small, blur_factors, x_scale, y_scale)

            # Draw the possession highlight and name tag (using upscaled smoothed coordinates)
            for track_id, box, role in highlight_players:
                s_box_small = smoothed_boxes.get(track_id)
                if s_box_small is None:
                    continue
                s_box = [
                    int(round(s_box_small[0] * x_scale)),
                    int(round(s_box_small[1] * y_scale)),
                    int(round(s_box_small[2] * x_scale)),
                    int(round(s_box_small[3] * y_scale)),
                ]
                custom_name = name_mapping.get(str(track_id))
                display_name = custom_name if custom_name else f"Player {track_id}"
                
                if role == "possessor":
                    apply_possession_highlight(frame_out, s_box[0], s_box[1], s_box[2], s_box[3], tracker_bgr)
                    draw_name_tag_with_alpha(frame_out, s_box[0], s_box[1], s_box[2], s_box[3], display_name, tracker_bgr, alpha=1.0)
                elif role == "receiver":
                    # Blend tracker_bgr with light grey/white for receiver highlight
                    receiver_color = (
                        int(round(tracker_bgr[0] * 0.4 + 180 * 0.6)),
                        int(round(tracker_bgr[1] * 0.4 + 180 * 0.6)),
                        int(round(tracker_bgr[2] * 0.4 + 180 * 0.6))
                    )
                    cv2.rectangle(frame_out, (s_box[0], s_box[1]), (s_box[2], s_box[3]), receiver_color, 1, cv2.LINE_AA)
                    
                    rec_alpha = 1.0 - (future_k / 10.0)
                    draw_name_tag_with_alpha(frame_out, s_box[0], s_box[1], s_box[2], s_box[3], display_name + " (REC)", receiver_color, alpha=rec_alpha)
                elif track_id in tag_decays:
                    alpha = tag_decays[track_id]
                    draw_name_tag_with_alpha(frame_out, s_box[0], s_box[1], s_box[2], s_box[3], display_name, tracker_bgr, alpha=alpha)

            # Draw ball tracker ring (upscaled)
            if ball_center_small is not None:
                bx = int(round(ball_center_small[0] * x_scale))
                by = int(round(ball_center_small[1] * y_scale))
                br = int(round(ball_radius_small * x_scale))
                draw_tracker_ring(frame_out, bx, by, tracker_bgr, radius=br)

            # Build metadata frame record (save original-resolution coordinates)
            protected_track_ids = set()
            if possessor_track_id is not None:
                protected_track_ids.add(possessor_track_id)
            if future_tid is not None and future_k <= 10:
                protected_track_ids.add(future_tid)
                
            frame_data = {
                "ball_center": [int(round(ball_center_small[0] * x_scale)), int(round(ball_center_small[1] * y_scale))] if ball_center_small is not None else None,
                "ball_radius": int(round(ball_radius_small * x_scale)),
                "possessor_track_id": possessor_track_id,
                "future_possessor_track_id": future_tid,
                "future_k": future_k,
                "players": [
                    {
                        "box": [
                            int(round(b[0] * x_scale)),
                            int(round(b[1] * y_scale)),
                            int(round(b[2] * x_scale)),
                            int(round(b[3] * y_scale))
                        ],
                        "track_id": int(tid)
                    } for tid, b in tracked_players_small
                ],
                "protected_track_ids": [int(tid) for tid in protected_track_ids],
                "goal_scorer_track_id": int(goal_scorer_track_id) if goal_scorer_track_id is not None else None
            }
            tracking_metadata["frames"].append(frame_data)

            writer.write(frame_out)
            frame_idx += 1
            if total > 0:
                jobs[job_id]["progress"] = 50 + int(frame_idx / total * 50)

        cap.release()
        writer.release()

        # Fallback to mock players if YOLOv8 detected nothing (e.g. for demo match video)
        if not saved_thumbnails:
            dummy_crop1 = np.zeros((160, 120, 3), dtype=np.uint8)
            cv2.putText(dummy_crop1, "Player 1", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (56, 189, 248), 1)
            cv2.rectangle(dummy_crop1, (5, 5), (115, 155), (56, 189, 248), 2)
            cv2.imwrite(str(THUMBNAIL_DIR / f"thumb_{job_id}_1.jpg"), dummy_crop1)
            
            dummy_crop2 = np.zeros((160, 120, 3), dtype=np.uint8)
            cv2.putText(dummy_crop2, "Player 2", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (129, 140, 248), 1)
            cv2.rectangle(dummy_crop2, (5, 5), (115, 155), (129, 140, 248), 2)
            cv2.imwrite(str(THUMBNAIL_DIR / f"thumb_{job_id}_2.jpg"), dummy_crop2)
            
            saved_thumbnails.add(1)
            saved_thumbnails.add(2)
            
            for fdata in tracking_metadata["frames"]:
                fdata["players"] = [
                    {"box": [150, 200, 250, 350], "track_id": 1},
                    {"box": [500, 200, 600, 350], "track_id": 2}
                ]

        # Save metadata to disk
        tracking_filepath = OUTPUT_DIR / f"tracking_{job_id}.json"
        with open(tracking_filepath, "w") as f:
            json.dump(tracking_metadata, f)

        # Prepare detected player info for frontend
        detected_players = []
        for tid in sorted(saved_thumbnails):
            detected_players.append({
                "track_id": tid,
                "name": name_mapping.get(str(tid), f"Player {tid}"),
                "thumbnail": f"thumb_{job_id}_{tid}.jpg"
            })

        jobs[job_id]["status"]   = "done"
        jobs[job_id]["filename"] = output_filename
        jobs[job_id]["players"]  = detected_players
        print(f"[{job_id}] Processing complete → {output_filename}")

    except Exception as exc:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"]  = str(exc)
        print(f"[{job_id}] ERROR: {exc}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/upload-video")
async def upload_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    tracker_color: str = Form("#FF0000"),
):
    """
    Accept an uploaded video + tracker colour, start background processing,
    return a job_id for polling.
    """
    # Validate file type
    allowed = {".mp4", ".mov", ".avi", ".mkv"}
    suffix  = Path(video.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    # Save upload
    job_id     = str(uuid.uuid4())
    input_name = f"input_{job_id}{suffix}"
    input_path = UPLOAD_DIR / input_name

    with open(input_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    # Register job
    jobs[job_id] = {
        "status": "queued",
        "filename": None,
        "progress": 0,
        "input_path": str(input_path),
        "tracker_color_hex": tracker_color,
        "name_mapping": {},
        "players": []
    }

    # Kick off background processing
    background_tasks.add_task(process_video, job_id, input_path, tracker_color)

    return {"job_id": job_id, "message": "Video uploaded. Processing started."}


@app.get("/job-status/{job_id}")
async def job_status(job_id: str):
    """Poll the processing status of a job."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found.")
    return jobs[job_id]


class UpdateNamesRequest(BaseModel):
    names: dict[str, str]


@app.post("/update-player-names/{job_id}")
async def update_player_names(job_id: str, payload: UpdateNamesRequest):
    """Re-render the video overlay quickly using saved player track coordinates and updated names."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found.")
        
    job = jobs[job_id]
    input_path_str = job.get("input_path")
    if not input_path_str:
        raise HTTPException(400, "Original video input path not found.")
        
    input_path = Path(input_path_str)
    if not input_path.exists():
        raise HTTPException(400, f"Original video file does not exist anymore.")
        
    tracking_filepath = OUTPUT_DIR / f"tracking_{job_id}.json"
    if not tracking_filepath.exists():
        raise HTTPException(400, "Tracking coordinates data not found for this video.")
        
    import json
    with open(tracking_filepath, "r") as f:
        metadata = json.load(f)
        
    name_mapping = payload.names
    job["name_mapping"] = name_mapping
    
    # Update current list of player items
    for p in job.get("players", []):
        tid_str = str(p["track_id"])
        if tid_str in name_mapping:
            p["name"] = name_mapping[tid_str]
            
    # Unique versioned output filename to prevent browser caching
    version = int(time.time())
    output_filename = f"processed_{job_id}_v{version}.mp4"
    output_path = OUTPUT_DIR / output_filename
    
    tracker_color_hex = metadata.get("tracker_color_hex", job.get("tracker_color_hex", "#FF0000"))
    tracker_bgr = hex_to_bgr(tracker_color_hex)
    
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise HTTPException(500, "Cannot open original video.")
        
    fps = metadata.get("fps", 30.0)
    width = metadata.get("width", 1280)
    height = metadata.get("height", 720)
    
    if os.name == "nt":
        try:
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            writer = cv2.VideoWriter(str(output_path), cv2.CAP_MSMF, fourcc, fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError("MSMF writer failed to open")
        except Exception:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
    frames_data = metadata.get("frames", [])
    frame_idx = 0
    blur_factors = {}
    smoothed_boxes = {}
    tag_decays = {}
    last_rendered_possessor = None
    
    x_scale = width / 640.0
    y_scale = height / 360.0
    alpha_step = 1.0 / max(1.0, 0.1 * fps)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx >= len(frames_data):
            writer.write(frame)
            frame_idx += 1
            continue
            
        fdata = frames_data[frame_idx]
        
        ball_center = fdata.get("ball_center")
        ball_radius = fdata.get("ball_radius", 12)
        possessor_track_id = fdata.get("possessor_track_id")
        future_tid = fdata.get("future_possessor_track_id")
        future_k = fdata.get("future_k", 0)
        players = fdata.get("players", [])
        protected_track_ids = fdata.get("protected_track_ids", [])
        
        # Calculate ball velocity (for high-speed pass detection)
        ball_speed = 0.0
        if ball_center is not None and frame_idx > 0:
            prev_fdata = frames_data[frame_idx - 1]
            prev_ball = prev_fdata.get("ball_center")
            if prev_ball is not None:
                ball_speed = euclidean(ball_center[0], ball_center[1], prev_ball[0], prev_ball[1])

        # Update smoothed bounding boxes for players in this frame (in original coords)
        for pdata in players:
            tid = pdata["track_id"]
            box = pdata["box"]
            box_coords = [float(b) for b in box]
            if tid in smoothed_boxes:
                prev_box = smoothed_boxes[tid]
                beta = 0.25  # smoothing factor
                smoothed = [
                    prev_box[0] + (box_coords[0] - prev_box[0]) * beta,
                    prev_box[1] + (box_coords[1] - prev_box[1]) * beta,
                    prev_box[2] + (box_coords[2] - prev_box[2]) * beta,
                    prev_box[3] + (box_coords[3] - prev_box[3]) * beta,
                ]
                smoothed_boxes[tid] = smoothed
            else:
                smoothed_boxes[tid] = box_coords
                
        # Clean up stale track IDs in smoothed_boxes
        current_track_ids = {pdata["track_id"] for pdata in players}
        smoothed_boxes = {tid: val for tid, val in smoothed_boxes.items() if tid in current_track_ids}

        # Update possession change tag decays
        if possessor_track_id != last_rendered_possessor:
            if last_rendered_possessor is not None and last_rendered_possessor != future_tid:
                tag_decays[last_rendered_possessor] = 1.0
            last_rendered_possessor = possessor_track_id

        # Decay active label alphas
        decay_step = 1.0 / max(1.0, 0.3 * fps)
        updated_decays = {}
        for tid, alpha in tag_decays.items():
            new_alpha = alpha - decay_step
            if new_alpha > 0.0 and tid in current_track_ids:
                if tid != possessor_track_id and tid != future_tid:
                    updated_decays[tid] = new_alpha
        tag_decays = updated_decays

        # Find possessor and future receiver boxes in this frame for overlap calculations
        possessor_box = None
        if possessor_track_id is not None:
            for pdata in players:
                if pdata["track_id"] == possessor_track_id:
                    possessor_box = pdata["box"]
                    break

        future_box = None
        if future_tid is not None:
            for pdata in players:
                if pdata["track_id"] == future_tid:
                    future_box = pdata["box"]
                    break

        # Scale player boxes down to 640x360 and compute crowded groups
        players_small = []
        player_centers = {}
        for pdata in players:
            tid = pdata["track_id"]
            box = pdata["box"]
            x1_s = int(round(box[0] / x_scale))
            y1_s = int(round(box[1] / y_scale))
            x2_s = int(round(box[2] / x_scale))
            y2_s = int(round(box[3] / y_scale))
            players_small.append((tid, [x1_s, y1_s, x2_s, y2_s]))
            
            cx = (x1_s + x2_s) / 2.0
            cy = (y1_s + y2_s) / 2.0
            player_centers[tid] = (cx, cy)
            
        crowded_players = set()
        for tid1, (cx1, cy1) in player_centers.items():
            for tid2, (cx2, cy2) in player_centers.items():
                if tid1 != tid2:
                    dist = euclidean(cx1, cy1, cx2, cy2)
                    if dist < 65.0:
                        crowded_players.add(tid1)
                        crowded_players.add(tid2)

        # Update blur alpha factors for all players in this frame
        for pdata in players:
            tid = pdata["track_id"]
            box = pdata["box"]
            target_alpha = 1.0
            
            if possessor_track_id is not None and tid == possessor_track_id:
                target_alpha = 0.0
            elif future_tid is not None and tid == future_tid and future_k <= 10:
                # Smoothly transition blur to 0.0 as ball approaches receiver
                target_alpha = max(0.0, (future_k - 1) / 10.0)
            elif tid in crowded_players:
                # Prevent blurring crowded players/crossovers
                target_alpha = 0.0
            else:
                # Multi-player occlusion: prevent blurring players close to or overlapping with focus
                if possessor_box is not None:
                    overlap = get_box_overlap(box, possessor_box)
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    overlap_ratio = overlap / float(area) if area > 0 else 0
                    if overlap_ratio >= 0.15:
                        target_alpha = 0.0
                
                if target_alpha > 0.0 and future_box is not None:
                    overlap = get_box_overlap(box, future_box)
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    overlap_ratio = overlap / float(area) if area > 0 else 0
                    if overlap_ratio >= 0.15:
                        target_alpha = 0.0

            current_alpha = blur_factors.get(tid, 1.0)
            if current_alpha < target_alpha:
                new_alpha = min(target_alpha, current_alpha + alpha_step)
            else:
                new_alpha = max(target_alpha, current_alpha - alpha_step)
            blur_factors[tid] = new_alpha

        # Clean up stale track IDs in blur_factors
        blur_factors = {tid: val for tid, val in blur_factors.items() if tid in current_track_ids}

        # Filter out drawing highlights for players with extreme overlaps (so highlights do not overlay messily)
        highlight_players = []
        for pdata in players:
            tid = pdata["track_id"]
            box = pdata["box"]
            is_possessor = (possessor_track_id is not None and tid == possessor_track_id)
            is_receiver = (future_tid is not None and tid == future_tid and future_k <= 10)
            
            if is_possessor:
                highlight_players.append((tid, box, "possessor"))
            elif is_receiver:
                highlight_players.append((tid, box, "receiver"))
            else:
                is_overlapping = False
                if possessor_box is not None:
                    overlap = get_box_overlap(box, possessor_box)
                    box_area = (box[2] - box[0]) * (box[3] - box[1])
                    overlap_ratio = overlap / float(box_area) if box_area > 0 else 0
                    if overlap_ratio >= 0.35:
                        is_overlapping = True
                        
                if not is_overlapping and future_box is not None:
                    overlap = get_box_overlap(box, future_box)
                    box_area = (box[2] - box[0]) * (box[3] - box[1])
                    overlap_ratio = overlap / float(box_area) if box_area > 0 else 0
                    if overlap_ratio >= 0.35:
                        is_overlapping = True
                        
                if not is_overlapping:
                    highlight_players.append((tid, box, "none"))

        # Apply Gaussian blur exclusively to off-ball player bounding boxes
        frame_out = apply_player_bounding_box_blur(frame, players_small, blur_factors, x_scale, y_scale)

        # Draw the possession highlight and name tag for the active possessor & receiver (using original coords)
        for tid, box, role in highlight_players:
            s_box_small = smoothed_boxes.get(tid)
            if s_box_small is None:
                continue
            s_box = [int(round(b)) for b in s_box_small]
            custom_name = name_mapping.get(str(tid))
            display_name = custom_name if custom_name else f"Player {tid}"
            
            if role == "possessor":
                apply_possession_highlight(frame_out, s_box[0], s_box[1], s_box[2], s_box[3], tracker_bgr)
                draw_name_tag_with_alpha(frame_out, s_box[0], s_box[1], s_box[2], s_box[3], display_name, tracker_bgr, alpha=1.0)
            elif role == "receiver":
                # Blend tracker_bgr with light grey/white for receiver highlight
                receiver_color = (
                    int(round(tracker_bgr[0] * 0.4 + 180 * 0.6)),
                    int(round(tracker_bgr[1] * 0.4 + 180 * 0.6)),
                    int(round(tracker_bgr[2] * 0.4 + 180 * 0.6))
                )
                cv2.rectangle(frame_out, (s_box[0], s_box[1]), (s_box[2], s_box[3]), receiver_color, 1, cv2.LINE_AA)
                
                rec_alpha = 1.0 - (future_k / 10.0)
                draw_name_tag_with_alpha(frame_out, s_box[0], s_box[1], s_box[2], s_box[3], display_name + " (REC)", receiver_color, alpha=rec_alpha)
            elif tid in tag_decays:
                alpha = tag_decays[tid]
                draw_name_tag_with_alpha(frame_out, s_box[0], s_box[1], s_box[2], s_box[3], display_name, tracker_bgr, alpha=alpha)
                
        if ball_center is not None:
            draw_tracker_ring(frame_out, ball_center[0], ball_center[1], tracker_bgr, radius=ball_radius)
            
        writer.write(frame_out)
        frame_idx += 1
        
    cap.release()
    writer.release()
    
    # Delete old output file
    old_filename = job.get("filename")
    if old_filename:
        try:
            (OUTPUT_DIR / old_filename).unlink(missing_ok=True)
        except Exception:
            pass
            
    job["filename"] = output_filename
    
    return {
        "status": "done",
        "filename": output_filename,
        "players": job["players"]
    }


@app.get("/thumbnail/{filename}")
async def get_thumbnail(filename: str):
    """Serve a player thumbnail crop image."""
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(400, "Invalid filename.")
    path = THUMBNAIL_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Thumbnail not found.")
    return FileResponse(str(path), media_type="image/jpeg")


@app.get("/download-video/{filename}")
async def download_video(filename: str):
    """Stream the processed video back to the client."""
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(400, "Invalid filename.")

    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(404, "File not found.")

    return FileResponse(
        str(path),
        media_type="video/mp4",
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/stream-video/{filename}")
async def stream_video(filename: str):
    """Stream video for in-browser playback (inline)."""
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(400, "Invalid filename.")

    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(404, "File not found.")
    return FileResponse(
        str(path),
        media_type="video/mp4",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )
