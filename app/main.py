import cv2
import mediapipe as mp
import numpy as np
import time
import os

WINDOW_NAME = "Posture Detection"

# Posture thresholds (degrees from vertical)
NECK_ANGLE_THRESHOLD = 15  # forward head detection
TORSO_ANGLE_THRESHOLD = 15  # slouching detection

# Right-side landmark indices
EAR = 8
SHOULDER = 12
HIP = 24

# Pose connections for drawing skeleton
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),  # Face
    (0, 4), (4, 5), (5, 6), (6, 8),  # Face
    (9, 10),  # Mouth
    (11, 12),  # Shoulders
    (11, 13), (13, 15),  # Left arm
    (12, 14), (14, 16),  # Right arm
    (11, 23), (12, 24),  # Torso
    (23, 24),  # Hips
    (23, 25), (25, 27),  # Left leg
    (24, 26), (26, 28),  # Right leg
]


def analyze_posture(landmarks):
    """
    Analyze posture from side view by measuring deviation from vertical.

    Neck angle: How far ear is forward/back of shoulder
    Torso angle: How far shoulder is forward/back of hip
    """
    ear = landmarks[EAR]
    shoulder = landmarks[SHOULDER]
    hip = landmarks[HIP]

    # Calculate neck inclination from vertical
    # atan2(horizontal_offset, vertical_distance) gives angle from vertical
    # Positive = ear is forward of shoulder
    neck_inclination = np.degrees(np.arctan2(
        ear.x - shoulder.x,    # horizontal offset
        shoulder.y - ear.y     # vertical distance (y increases downward)
    ))
    neck_angle = abs(neck_inclination)

    # Calculate torso inclination from vertical
    # Positive = shoulder is forward of hip
    torso_inclination = np.degrees(np.arctan2(
        shoulder.x - hip.x,
        hip.y - shoulder.y
    ))
    torso_angle = abs(torso_inclination)

    is_good = True
    alerts = []

    if neck_angle > NECK_ANGLE_THRESHOLD:
        is_good = False
        alerts.append(f"Forward head: {neck_angle:.0f} deg")

    if torso_angle > TORSO_ANGLE_THRESHOLD:
        is_good = False
        alerts.append(f"Slouching: {torso_angle:.0f} deg")

    return is_good, neck_angle, torso_angle, alerts


def draw_landmarks(frame, landmarks):
    """Draw pose landmarks and connections on frame."""
    h, w = frame.shape[:2]

    # Draw connections
    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            if start.visibility > 0.5 and end.visibility > 0.5:
                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))
                cv2.line(frame, start_point, end_point, (255, 0, 0), 2)

    # Draw landmark points
    for i, landmark in enumerate(landmarks):
        if landmark.visibility > 0.5:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            # Highlight key posture landmarks
            if i in [EAR, SHOULDER, HIP]:
                cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)
            else:
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)


def main():
    # Get model path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "pose_landmarker_lite.task")

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Download it with:")
        print("curl -o pose_landmarker_lite.task -L https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task")
        return

    # Initialize MediaPipe Pose Landmarker (new Tasks API)
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    landmarker = PoseLandmarker.create_from_options(options)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    print("Posture detection started. Sit sideways to the camera. Press 'q' to quit.")

    cv2.moveWindow(WINDOW_NAME, 100, 100)

    # Track timestamps for VIDEO mode
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Calculate timestamp in milliseconds
        timestamp_ms = int((time.time() - start_time) * 1000)

        # Process with MediaPipe Pose Landmarker
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        if results.pose_landmarks and len(results.pose_landmarks) > 0:
            landmarks = results.pose_landmarks[0]

            # Check if key landmarks are visible
            if (landmarks[EAR].visibility > 0.5 and
                landmarks[SHOULDER].visibility > 0.5 and
                landmarks[HIP].visibility > 0.5):

                # Analyze posture
                is_good, neck_angle, torso_angle, alerts = analyze_posture(landmarks)

                # Display posture status
                if is_good:
                    status_text = "GOOD POSTURE"
                    color = (0, 255, 0)  # Green
                else:
                    status_text = "BAD POSTURE"
                    color = (0, 0, 255)  # Red

                cv2.putText(frame, status_text, (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

                cv2.putText(frame, f"Neck: {neck_angle:.0f} deg  Torso: {torso_angle:.0f} deg", (20, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Display alerts
                for i, alert in enumerate(alerts):
                    cv2.putText(frame, alert, (20, 130 + i * 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Position yourself sideways", (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Draw pose landmarks
            draw_landmarks(frame, landmarks)

        cv2.imshow(WINDOW_NAME, frame)

        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
