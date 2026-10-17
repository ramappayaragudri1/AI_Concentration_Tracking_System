import sys
# Prevent conflict between global tensorflow and mediapipe/protobuf
sys.modules['tensorflow'] = None

import cv2
import mediapipe as mp
import numpy as np
import time
import os
import csv
from datetime import datetime
from collections import deque
from mediapipe.python.solutions.drawing_utils import DrawingSpec

# Native Windows sound support for distraction alerts
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

score_history = deque(maxlen=10)
recent_scores = deque(maxlen=50)
distraction = 0
total_blinks = 0
blink_state = False
last_beep_time = 0
session_start_time = time.time()
session_logs = []

def play_alert_sound(frequency=600, duration=250):
    """Play non-blocking alert sound on Windows"""
    global last_beep_time
    now = time.time()
    if HAS_WINSOUND and (now - last_beep_time > 3.0):
        try:
            winsound.Beep(frequency, duration)
            last_beep_time = now
        except Exception:
            pass

def eye_aspect_ratio(landmarks, eye_points, image_w, image_h, frame):
    p = []
    for idx in eye_points:
        lm = landmarks[idx]
        x, y = int(lm.x * image_w), int(lm.y * image_h)
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1) 
        p.append((x, y))

    A = np.linalg.norm(np.array(p[1]) - np.array(p[5]))
    B = np.linalg.norm(np.array(p[2]) - np.array(p[4]))
    C = np.linalg.norm(np.array(p[0]) - np.array(p[3]))
    ear = (A + B) / (2.0 * C + 1e-6)
    return ear

def is_blinking(ear, threshold=0.20):
    return ear < threshold

def get_head_pose_score(landmarks, image_w, image_h):
    """
    Direct facial geometry check:
    Measures nose position relative to the center of the face (between eyes).
    Guarantees 1.0 (100%) when facing your display.
    """
    nose = landmarks[1]
    l_eye = landmarks[33]
    r_eye = landmarks[263]

    # Horizontal center between both eyes
    mid_x = (l_eye.x + r_eye.x) / 2.0
    face_width = abs(r_eye.x - l_eye.x) + 1e-6
    offset = abs(nose.x - mid_x) / face_width

    # When facing the camera/screen directly, offset is < 0.20
    if offset < 0.20:
        return 1.0
    elif offset < 0.35:
        return 0.7
    return 0.0

def get_gaze_score(landmarks, image_w, image_h):
    """
    Eye-relative iris tracking:
    Works accurately regardless of where you sit in front of the camera.
    """
    left_iris = landmarks[468]
    right_iris = landmarks[473]
    l_outer = landmarks[33]
    l_inner = landmarks[133]
    r_inner = landmarks[362]
    r_outer = landmarks[263]

    l_width = abs(l_inner.x - l_outer.x) + 1e-6
    r_width = abs(r_outer.x - r_inner.x) + 1e-6

    l_ratio = (left_iris.x - min(l_outer.x, l_inner.x)) / l_width
    r_ratio = (right_iris.x - min(r_inner.x, r_outer.x)) / r_width
    avg_ratio = (l_ratio + r_ratio) / 2.0

    # Looking at screen -> iris is naturally centered [0.30 - 0.70]
    if 0.30 <= avg_ratio <= 0.70:
        return 1.0
    elif 0.20 <= avg_ratio <= 0.80:
        return 0.75
    return 0.0

def compute_concentration_score(gaze, head_pose, blink):
    """Formula: 40% Gaze + 40% Head Pose + 20% Eyes Open = 100%"""
    score = 0.4 * gaze + 0.4 * head_pose + 0.2 * (0 if blink else 1)
    return round(score * 100, 2)

def print_session_report(start_time, total_blinks_cnt, distraction_cnt, scores):
    """Print ASCII summary and save CSV report"""
    duration = time.time() - start_time
    mins = int(duration // 60)
    secs = int(duration % 60)
    avg_score = round(float(np.mean(scores)), 1) if scores else 0.0
    bpm = round((total_blinks_cnt / (duration / 60.0)), 1) if duration > 5 else 0

    print("\n" + "=" * 60)
    print("       🎯 AI CONCENTRATION TRACKER - SESSION SUMMARY")
    print("=" * 60)
    print(f"  Total Duration:         {mins:02d}m {secs:02d}s ({duration:.1f} seconds)")
    print(f"  Average Concentration:  {avg_score}%")
    print(f"  Total Blinks:           {total_blinks_cnt} ({bpm} blinks/min)")
    print(f"  Distraction Count:      {distraction_cnt}")
    print(f"  Focus Quality:          {'EXCELLENT' if avg_score >= 75 else 'GOOD' if avg_score >= 50 else 'NEEDS IMPROVEMENT'}")
    print("=" * 60)

    # Save to CSV
    os.makedirs("reports", exist_ok=True)
    report_filename = os.path.join("reports", f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    try:
        with open(report_filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "ConcentrationScore", "EAR", "GazeScore", "HeadPoseScore", "IsBlink"])
            writer.writerows(session_logs)
        print(f"  Detailed session CSV saved to: {report_filename}\n")
    except Exception as e:
        print(f"  Could not save CSV report: {e}\n")

# Main Application
def main():
    global distraction, total_blinks, blink_state

    print("=" * 60)
    print("  Starting AI Concentration Tracker (Big Screen Dashboard)...")
    print("  Controls: Press 'q' on the video window to quit & export report.")
    print("=" * 60)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("\n❌ Error: Could not open webcam.")
        return

    # Widescreen Big Dashboard Window Setup
    WINDOW_TITLE = "AI Concentration Tracker - Dashboard"
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, 1280, 720)

    DASH_WIDTH = 1280
    DASH_HEIGHT = 720
    VIDEO_W = 900
    VIDEO_H = 675

    prev_time = time.time()
    fps = 30.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        fps = 1.0 / (now - prev_time + 1e-6)
        prev_time = now

        image_h, image_w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        smooth_score = 0
        blink = False
        ear_val = 0.28
        gaze_score = 0.0
        head_score = 0.0

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame, 
                    face_landmarks, 
                    mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=DrawingSpec(color=(0, 220, 100), thickness=1, circle_radius=1),
                    connection_drawing_spec=DrawingSpec(color=(0, 150, 255), thickness=1)
                )

                landmarks = face_landmarks.landmark
                left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, image_w, image_h, frame)
                right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, image_w, image_h, frame)
                ear_val = (left_ear + right_ear) / 2.0

                blink = is_blinking(ear_val)
                if blink:
                    if not blink_state:
                        total_blinks += 1
                        blink_state = True
                else:
                    blink_state = False

                gaze_score = get_gaze_score(landmarks, image_w, image_h)
                head_score = get_head_pose_score(landmarks, image_w, image_h)
                concentration = compute_concentration_score(gaze_score, head_score, blink)

                score_history.append(concentration)
                smooth_score = int(np.mean(score_history))
                recent_scores.append(smooth_score)

                if smooth_score < 40:
                    distraction += 1
                    if distraction % 60 == 0:  # ~2 sec
                        play_alert_sound(frequency=500, duration=250)
                else:
                    distraction = 0

                # Log data
                session_logs.append([
                    datetime.now().strftime("%H:%M:%S.%f")[:-3],
                    smooth_score,
                    round(ear_val, 2),
                    gaze_score,
                    head_score,
                    1 if blink else 0
                ])

        # ========================================================
        # BUILD WIDESCREEN BIG DASHBOARD (1280 x 720)
        # ========================================================
        dashboard = np.full((DASH_HEIGHT, DASH_WIDTH, 3), (16, 20, 28), dtype=np.uint8)

        # 1. Resize and place camera video on the left
        resized_video = cv2.resize(frame, (VIDEO_W, VIDEO_H))
        y_offset = (DASH_HEIGHT - VIDEO_H) // 2
        x_offset = 20
        dashboard[y_offset:y_offset + VIDEO_H, x_offset:x_offset + VIDEO_W] = resized_video
        cv2.rectangle(dashboard, (x_offset, y_offset), (x_offset + VIDEO_W, y_offset + VIDEO_H), (50, 60, 80), 2)

        # Video HUD Overlay (FPS + Exit prompt)
        cv2.putText(dashboard, f"FPS: {fps:.1f}", (x_offset + 15, y_offset + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 240, 255), 2, cv2.LINE_AA)
        
        if blink:
            cv2.putText(dashboard, "BLINKING", (x_offset + 15, y_offset + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 150, 255), 2, cv2.LINE_AA)

        # 2. Right Side Big Analytics Panel
        panel_x = x_offset + VIDEO_W + 20
        panel_w = DASH_WIDTH - panel_x - 20
        panel_y = y_offset
        panel_h = VIDEO_H

        # Panel Background Card
        cv2.rectangle(dashboard, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (24, 30, 42), -1)
        cv2.rectangle(dashboard, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (60, 75, 100), 2)

        # Title
        cv2.putText(dashboard, "CONCENTRATION", (panel_x + 20, panel_y + 40),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(dashboard, "MONITOR", (panel_x + 20, panel_y + 70),
                    cv2.FONT_HERSHEY_DUPLEX, 0.95, (0, 240, 255), 2, cv2.LINE_AA)

        # Elapsed Session Time
        elapsed = time.time() - session_start_time
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        cv2.putText(dashboard, f"Session: {mins:02d}:{secs:02d}", (panel_x + 20, panel_y + 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 190, 205), 1, cv2.LINE_AA)

        # Big Concentration Score Percentage Display
        score_col = (0, 220, 100) if smooth_score >= 70 else (0, 180, 255) if smooth_score >= 40 else (0, 60, 255)
        cv2.putText(dashboard, f"{smooth_score}%", (panel_x + 20, panel_y + 200),
                    cv2.FONT_HERSHEY_DUPLEX, 2.5, score_col, 4, cv2.LINE_AA)

        # Big Concentration Bar
        bar_x = panel_x + 20
        bar_y = panel_y + 230
        bar_w = panel_w - 40
        bar_h = 28
        cv2.rectangle(dashboard, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 48, 65), -1)
        fill_w = int((smooth_score / 100.0) * bar_w)
        if fill_w > 0:
            cv2.rectangle(dashboard, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), score_col, -1)
        cv2.rectangle(dashboard, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (120, 140, 170), 2)

        # Focus Status Badge
        status_text = "FOCUSED" if smooth_score >= 40 else "DISTRACTED"
        badge_bg = (0, 120, 50) if smooth_score >= 40 else (0, 30, 140)
        cv2.rectangle(dashboard, (panel_x + 20, panel_y + 285), (panel_x + panel_w - 20, panel_y + 335), badge_bg, -1)
        cv2.rectangle(dashboard, (panel_x + 20, panel_y + 285), (panel_x + panel_w - 20, panel_y + 335), score_col, 2)
        cv2.putText(dashboard, status_text, (panel_x + 45, panel_y + 322),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

        # Real-time Trend Graph inside panel
        graph_x = panel_x + 20
        graph_y = panel_y + 370
        graph_w = panel_w - 40
        graph_h = 140

        cv2.rectangle(dashboard, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (18, 22, 32), -1)
        cv2.rectangle(dashboard, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (60, 70, 90), 1)
        cv2.putText(dashboard, "REAL-TIME TREND (30s)", (graph_x + 10, graph_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 150, 170), 1, cv2.LINE_AA)

        # Draw sparkline points
        if len(recent_scores) > 1:
            pts = []
            dx = graph_w / float(recent_scores.maxlen)
            for i, s in enumerate(recent_scores):
                px = int(graph_x + i * dx)
                py = int(graph_y + graph_h - (s / 100.0) * (graph_h - 35) - 8)
                pts.append((px, py))
            cv2.polylines(dashboard, [np.array(pts, dtype=np.int32)], False, (0, 240, 255), 2, cv2.LINE_AA)

        # Bottom Controls Instruction
        cv2.putText(dashboard, "Press 'q' to Quit & Export", (panel_x + 25, panel_y + panel_h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 130, 150), 1, cv2.LINE_AA)

        # Show Dashboard
        cv2.imshow(WINDOW_TITLE, dashboard)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Print summary and export CSV
    all_scores = [log[1] for log in session_logs]
    print_session_report(session_start_time, total_blinks, distraction, all_scores)

if __name__ == "__main__":
    main()