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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_private_network=True,
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


def blur_player_region(frame: np.ndarray,
                        x1: int, y1: int, x2: int, y2: int,
                        protection_mask: np.ndarray | None = None) -> None:
    """Gaussian-blur a player's region in-place, respecting the protection mask if provided."""
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return

    roi = frame[y1:y2, x1:x2]
    blurred = cv2.GaussianBlur(roi, (15, 15), 0)

    if protection_mask is not None:
        roi_mask = protection_mask[y1:y2, x1:x2]
        mask_3d = np.expand_dims(roi_mask, axis=2)
        frame[y1:y2, x1:x2] = (roi * (1 - mask_3d) + blurred * mask_3d).astype(np.uint8)
    else:
        frame[y1:y2, x1:x2] = blurred


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


class IoUTracker:
    def __init__(self, iou_threshold=0.3, max_lost_frames=30):
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.next_id = 1
        self.tracks = {}  # id -> {"box": (x1, y1, x2, y2), "lost_frames": int}

    def update(self, detections):
        updated_tracks = {}
        matched_detections = set()

        for track_id, track_data in self.tracks.items():
            if track_data["lost_frames"] > self.max_lost_frames:
                continue

            best_iou = 0
            best_det_idx = -1

            for idx, det in enumerate(detections):
                if idx in matched_detections:
                    continue
                iou = self.calculate_iou(track_data["box"], det)
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = idx

            if best_iou >= self.iou_threshold:
                updated_tracks[track_id] = {
                    "box": detections[best_det_idx],
                    "lost_frames": 0
                }
                matched_detections.add(best_det_idx)
            else:
                updated_tracks[track_id] = {
                    "box": track_data["box"],
                    "lost_frames": track_data["lost_frames"] + 1
                }

        for idx, det in enumerate(detections):
            if idx not in matched_detections:
                updated_tracks[self.next_id] = {
                    "box": det,
                    "lost_frames": 0
                }
                self.next_id += 1

        self.tracks = updated_tracks
        
        active_detections = []
        for track_id, track_data in self.tracks.items():
            if track_data["lost_frames"] == 0:
                active_detections.append((track_id, track_data["box"]))
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
        ball_velocity_x = 0
        ball_velocity_y = 0
        ball_lost_frames = 0

        # Player tracker
        player_tracker = IoUTracker()
        saved_thumbnails = set()

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

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=0.1, verbose=False)[0]

            players: list[tuple[int, int, int, int]] = []
            ball_candidates: list[tuple[tuple[int, int], int, float]] = []

            for box in results.boxes:
                cls_id     = int(box.cls[0])
                conf       = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if cls_id == 0 and conf > 0.35:          # player
                    players.append((x1, y1, x2, y2))
                elif cls_id == 32:                       # ball
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    r = max(6, (x2 - x1 + y2 - y1) // 4)
                    ball_candidates.append(((cx, cy), r, conf))

            # Update tracked players
            tracked_players = player_tracker.update(players)

            # Crop player thumbnails on first occurrence
            for tid, (x1, y1, x2, y2) in tracked_players:
                if tid not in saved_thumbnails:
                    h_f, w_f = frame.shape[:2]
                    px1, py1, px2, py2 = max(0, x1), max(0, y1), min(w_f, x2), min(h_f, y2)
                    if px2 > px1 and py2 > py1:
                        crop = frame[py1:py2, px1:px2]
                        crop_resized = cv2.resize(crop, (120, 160))
                        thumb_filename = f"thumb_{job_id}_{tid}.jpg"
                        cv2.imwrite(str(THUMBNAIL_DIR / thumb_filename), crop_resized)
                        saved_thumbnails.add(tid)

            # ----------------------------------------------------------------
            # Ball tracking and projection
            # ----------------------------------------------------------------
            current_ball = None
            current_radius = 12
            predicted_ball = None
            if last_ball_center is not None and ball_lost_frames < 15:
                pred_x = last_ball_center[0] + ball_velocity_x
                pred_y = last_ball_center[1] + ball_velocity_y
                predicted_ball = (pred_x, pred_y)

            best_score = -1
            best_candidate = None
            best_radius = 12

            for (cx, cy), r, conf in ball_candidates:
                if predicted_ball is not None:
                    dist = euclidean(cx, cy, predicted_ball[0], predicted_ball[1])
                    if dist < 120:
                        score = conf * (1.0 - (dist / 120.0) * 0.4)
                        if score > best_score:
                            best_score = score
                            best_candidate = (cx, cy)
                            best_radius = r
                else:
                    if conf > 0.15:
                        if conf > best_score:
                            best_score = conf
                            best_candidate = (cx, cy)
                            best_radius = r

            if best_candidate is not None:
                if last_ball_center is not None:
                    ball_velocity_x = best_candidate[0] - last_ball_center[0]
                    ball_velocity_y = best_candidate[1] - last_ball_center[1]
                    max_vel = 80
                    ball_velocity_x = max(-max_vel, min(max_vel, ball_velocity_x))
                    ball_velocity_y = max(-max_vel, min(max_vel, ball_velocity_y))

                current_ball = best_candidate
                current_radius = best_radius
                last_ball_center = best_candidate
                last_ball_radius = best_radius
                ball_lost_frames = 0
            else:
                if last_ball_center is not None and ball_lost_frames < 8:
                    projected_x = int(last_ball_center[0] + ball_velocity_x)
                    projected_y = int(last_ball_center[1] + ball_velocity_y)
                    ball_velocity_x *= 0.85
                    ball_velocity_y *= 0.85
                    current_ball = (projected_x, projected_y)
                    current_radius = last_ball_radius
                    last_ball_center = current_ball
                    ball_lost_frames += 1
                else:
                    current_ball = None
                    last_ball_center = None
                    last_ball_radius = 12
                    ball_lost_frames = 0
                    ball_velocity_x = 0
                    ball_velocity_y = 0

            ball_center = current_ball
            ball_radius = current_radius

            # ----------------------------------------------------------------
            # Ball possession detection
            # ----------------------------------------------------------------
            possessor_track_id = None

            if ball_center is not None and tracked_players:
                bx, by = ball_center
                min_dist = float("inf")
                closest_track_id = None
                
                for track_id, (x1, y1, x2, y2) in tracked_players:
                    fx = (x1 + x2) // 2
                    fy = y2
                    dist = euclidean(bx, by, fx, fy)
                    if dist < min_dist:
                        min_dist = dist
                        closest_track_id = track_id

                if min_dist <= POSSESSION_THRESHOLD_PX:
                    possessor_track_id = closest_track_id
                    active_possessor_track_id = closest_track_id
                    possessor_lost_frames = 0
                else:
                    if active_possessor_track_id is not None:
                        prev_box = None
                        for track_id, box in tracked_players:
                            if track_id == active_possessor_track_id:
                                prev_box = box
                                break

                        if prev_box is not None:
                            p_fx = (prev_box[0] + prev_box[2]) // 2
                            p_fy = prev_box[3]
                            dist_to_possessor = euclidean(bx, by, p_fx, p_fy)
                            if dist_to_possessor < 120:
                                possessor_track_id = active_possessor_track_id
                            else:
                                active_possessor_track_id = None
                        else:
                            active_possessor_track_id = None
            else:
                if active_possessor_track_id is not None and tracked_players:
                    still_present = any(track_id == active_possessor_track_id for track_id, _ in tracked_players)
                    if still_present and possessor_lost_frames < 45:
                        possessor_track_id = active_possessor_track_id
                        possessor_lost_frames += 1
                    else:
                        active_possessor_track_id = None
                        possessor_lost_frames = 0
                else:
                    active_possessor_track_id = None
                    possessor_lost_frames = 0

            # Update last possessor before shot
            if possessor_track_id is not None:
                last_possessor_track_id_before_shot = possessor_track_id

            # Goal scorer detection
            goal_scorer_active = False
            if ball_center is not None:
                bx, by = ball_center
                in_left_goal = (bx < width * 0.25) and (height * 0.2 < by < height * 0.8)
                in_right_goal = (bx > width * 0.75) and (height * 0.2 < by < height * 0.8)

                if (in_left_goal or in_right_goal) and last_possessor_track_id_before_shot is not None:
                    goal_scorer_track_id = last_possessor_track_id_before_shot
                    goal_scorer_frames_left = 180
                    last_possessor_track_id_before_shot = None

            if goal_scorer_frames_left > 0 and goal_scorer_track_id is not None:
                still_present = any(track_id == goal_scorer_track_id for track_id, _ in tracked_players)
                if still_present:
                    goal_scorer_active = True
                goal_scorer_frames_left -= 1
            else:
                goal_scorer_track_id = None
                goal_scorer_active = False

            # Identify protected player track IDs
            protected_track_ids = set()
            if possessor_track_id is not None:
                protected_track_ids.add(possessor_track_id)

            if ball_center is not None:
                bx, by = ball_center
                for track_id, (x1, y1, x2, y2) in tracked_players:
                    d_box = dist_to_box(bx, by, x1, y1, x2, y2)
                    if d_box <= 65.0:
                        protected_track_ids.add(track_id)

            if goal_scorer_active and goal_scorer_track_id is not None:
                protected_track_ids.add(goal_scorer_track_id)

            # Build metadata frame record
            frame_data = {
                "ball_center": ball_center,
                "ball_radius": ball_radius,
                "possessor_track_id": possessor_track_id,
                "players": [{"box": [int(x1), int(y1), int(x2), int(y2)], "track_id": int(tid)} for tid, (x1, y1, x2, y2) in tracked_players],
                "protected_track_ids": [int(tid) for tid in protected_track_ids],
                "goal_scorer_track_id": int(goal_scorer_track_id) if goal_scorer_track_id is not None else None
            }
            tracking_metadata["frames"].append(frame_data)

            # Create protection mask
            h_f, w_f = frame.shape[:2]
            protection_mask = np.ones((h_f, w_f), dtype=np.uint8)
            for track_id, (x1, y1, x2, y2) in tracked_players:
                if track_id in protected_track_ids:
                    x1_p, y1_p = max(0, x1), max(0, y1)
                    x2_p, y2_p = min(w_f, x2), min(h_f, y2)
                    protection_mask[y1_p:y2_p, x1_p:x2_p] = 0

            # Duplicate / overlap suppression
            possessor_box = None
            if possessor_track_id is not None:
                for track_id, box in tracked_players:
                    if track_id == possessor_track_id:
                        possessor_box = box
                        break

            filtered_players = []
            for track_id, box in tracked_players:
                if track_id == possessor_track_id:
                    filtered_players.append((track_id, box, True))
                else:
                    if possessor_box is not None:
                        overlap = get_box_overlap(box, possessor_box)
                        box_area = (box[2] - box[0]) * (box[3] - box[1])
                        overlap_ratio = overlap / float(box_area) if box_area > 0 else 0
                        if overlap_ratio >= 0.35:
                            continue
                    filtered_players.append((track_id, box, False))

            # Update possession durations and tag decays
            current_possessor = possessor_track_id
            if current_possessor is not None:
                possessor_durations[current_possessor] = possessor_durations.get(current_possessor, 0) + 1
                tag_decays[current_possessor] = 0
                
            for tid in list(possessor_durations.keys()):
                if tid != current_possessor:
                    duration = possessor_durations[tid]
                    if duration > 0:
                        decay_val = int(duration * 0.5)
                        decay_val = max(15, min(60, decay_val))
                        tag_decays[tid] = decay_val
                        possessor_durations[tid] = 0

            for track_id, box, is_possessor in filtered_players:
                is_protected = track_id in protected_track_ids

                if is_possessor:
                    apply_possession_highlight(frame, box[0], box[1], box[2], box[3], tracker_bgr)
                elif is_protected:
                    pass
                else:
                    height = box[3] - box[1]
                    width = box[2] - box[0]
                    x1_expanded = box[0] - int(width * 0.15)
                    x2_expanded = box[2] + int(width * 0.15)
                    y2_expanded = box[3] + int(height * 0.25)
                    blur_player_region(frame, x1_expanded, box[1], x2_expanded, y2_expanded, protection_mask)

                # Draw player name tag (if possessing OR during possession decay period)
                should_tag = is_possessor or (tag_decays.get(track_id, 0) > 0)
                if should_tag:
                    custom_name = name_mapping.get(str(track_id))
                    display_name = custom_name if custom_name else f"Player {track_id}"
                    draw_name_tag(frame, box[0], box[1], box[2], box[3], display_name, tracker_bgr)

            # Decrement active decays
            for tid in list(tag_decays.keys()):
                if tag_decays[tid] > 0:
                    tag_decays[tid] -= 1

            # Draw tracker ring
            if ball_center is not None:
                draw_tracker_ring(frame, ball_center[0], ball_center[1], tracker_bgr, radius=ball_radius)

            writer.write(frame)
            frame_idx += 1

            if total > 0:
                pct = int(frame_idx / total * 100)
                jobs[job_id]["progress"] = pct

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
        players = fdata.get("players", [])
        protected_track_ids = fdata.get("protected_track_ids", [])
        
        h_f, w_f = frame.shape[:2]
        protection_mask = np.ones((h_f, w_f), dtype=np.uint8)
        
        for pdata in players:
            tid = pdata["track_id"]
            if tid in protected_track_ids:
                x1, y1, x2, y2 = pdata["box"]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_f, x2), min(h_f, y2)
                protection_mask[y1:y2, x1:x2] = 0
                
        possessor_box = None
        if possessor_track_id is not None:
            for pdata in players:
                if pdata["track_id"] == possessor_track_id:
                    possessor_box = pdata["box"]
                    break
                    
        filtered_players = []
        for pdata in players:
            tid = pdata["track_id"]
            box = pdata["box"]
            if tid == possessor_track_id:
                filtered_players.append((tid, box, True))
            else:
                if possessor_box is not None:
                    overlap = get_box_overlap(box, possessor_box)
                    box_area = (box[2] - box[0]) * (box[3] - box[1])
                    overlap_ratio = overlap / float(box_area) if box_area > 0 else 0
                    if overlap_ratio >= 0.35:
                        continue
                filtered_players.append((tid, box, False))
                
        for tid, box, is_possessor in filtered_players:
            is_protected = tid in protected_track_ids
            
            if is_possessor:
                apply_possession_highlight(frame, box[0], box[1], box[2], box[3], tracker_bgr)
            elif is_protected:
                pass
            else:
                height = box[3] - box[1]
                width = box[2] - box[0]
                x1_expanded = box[0] - int(width * 0.15)
                x2_expanded = box[2] + int(width * 0.15)
                y2_expanded = box[3] + int(height * 0.25)
                blur_player_region(frame, x1_expanded, box[1], x2_expanded, y2_expanded, protection_mask)
                
            if is_possessor:
                custom_name = name_mapping.get(str(tid))
                display_name = custom_name if custom_name else f"Player {tid}"
                draw_name_tag(frame, box[0], box[1], box[2], box[3], display_name, tracker_bgr)
                
        if ball_center is not None:
            draw_tracker_ring(frame, ball_center[0], ball_center[1], tracker_bgr, radius=ball_radius)
            
        writer.write(frame)
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
