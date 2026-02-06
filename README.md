# Posture Detection

A real-time posture detection application that uses your webcam and Google's MediaPipe pose estimation to detect bad sitting posture — specifically forward head tilt and slouching — and plays audio alerts when you've been sitting badly for too long.

The user sits sideways to their camera. The app tracks key body landmarks (ear, shoulder, hip), calculates the angles between them, and determines whether posture deviates from a healthy upright position.

## Libraries

### `cv2` (OpenCV)

OpenCV is the computer vision library that handles everything the user actually sees. It captures frames from the webcam (`VideoCapture`), draws the skeleton overlay, posture status text, and angle readouts onto each frame, and displays the result in a window. It also handles user input — the app exits when you press `q` or close the window.

**Installed via:** `opencv-python`

### `mediapipe`

MediaPipe is Google's ML framework for on-device perception. This project uses its **Pose Landmarker** task, which takes a single 2D camera image and returns 33 body keypoints (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles, etc.) each with x/y/z coordinates and a visibility confidence score. This is the machine learning backbone of the entire application — without it, there is no pose detection.

### `numpy`

NumPy provides the math needed to turn raw landmark coordinates into meaningful posture measurements. Specifically, `np.arctan2` computes the angle between two points (how far forward the ear is relative to the shoulder, or the shoulder relative to the hip), and `np.degrees` converts that result from radians to human-readable degrees.

### `time`

The standard library `time` module serves two purposes:

1. **Timestamp generation** — MediaPipe's VIDEO running mode requires monotonically increasing timestamps for each frame. The app records a `start_time` and computes millisecond offsets from it.
2. **Alert timing** — tracks how long bad posture has been sustained (`BAD_POSTURE_DELAY`) and enforces a cooldown between consecutive audio alerts (`ALERT_COOLDOWN`).

### `os`

Used to resolve file paths relative to the script's own location. The model file (`pose_landmarker_lite.task`) and the `sounds/` directory both live alongside `main.py`, so `os.path.dirname(os.path.abspath(__file__))` gives a reliable base path regardless of where the script is invoked from.

### `subprocess`

Spawns the macOS `afplay` command as a non-blocking child process to play MP3 alert sounds. Using `subprocess.Popen` (rather than `subprocess.run`) means the audio plays in the background while the video loop continues uninterrupted — the webcam feed doesn't freeze while a sound is playing.

### `glob`

The standard library `glob` function discovers all `*.mp3` files in the `sounds/` directory at startup. This lets you add, remove, or rename sound files without changing any code — the app just picks up whatever MP3s are in the folder.

## Why the Pose Landmarker Model File Is Required

The file `pose_landmarker_lite.task` is a TensorFlow Lite model bundle that MediaPipe's Tasks API needs to perform pose estimation. Unlike older versions of MediaPipe (the "solutions" API) that downloaded models automatically behind the scenes, the newer Tasks API requires you to provide the model file explicitly.

This file contains a lightweight neural network trained to map a 2D camera image to 33 body keypoints. It runs entirely on-device — no internet connection or cloud API is needed at runtime. The "lite" variant is optimized for speed over accuracy, which is ideal for a real-time webcam application where low latency matters more than sub-pixel precision.

Without this file, the `PoseLandmarker.create_from_options()` call would fail and the app cannot function at all. There is no fallback — pose estimation is the entire purpose of the application.

**Download it with:**

```
curl -o app/pose_landmarker_lite.task -L https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
```

## Constants and Configuration

### Posture thresholds

```python
NECK_ANGLE_THRESHOLD = 15  # degrees from vertical
TORSO_ANGLE_THRESHOLD = 15
```

These define how many degrees of deviation from a perfectly upright position count as "bad posture." A neck angle above 15 degrees means the head is jutting forward; a torso angle above 15 degrees means the shoulders are slouching forward relative to the hips.

### Alert timing

```python
ALERT_COOLDOWN = 5    # seconds between alerts
BAD_POSTURE_DELAY = 3  # seconds of sustained bad posture before alerting
```

The app doesn't alert immediately when it detects bad posture — momentary shifts (reaching for something, looking down briefly) are ignored. The posture must be bad for 3 continuous seconds before the first alert plays. After an alert, no new alert will play for at least 5 seconds, even if posture remains bad.

### Landmark indices

```python
EAR = 8
LEFT_SHOULDER = 11
SHOULDER = 12
LEFT_HIP = 23
HIP = 24
```

MediaPipe's pose model returns 33 landmarks in a fixed order. These constants name the specific indices used for posture calculation. The right-side landmarks (ear index 8, right shoulder 12, right hip 24) are the primary measurement points since the user sits sideways. The left-side counterparts (11, 23) are used to detect whether the user is actually sideways.

### Sideways detection

```python
SHOULDER_SPREAD_THRESHOLD = 0.12
```

When the user faces the camera head-on, the left and right shoulders are far apart in the frame (high x-coordinate spread). When sideways, they overlap nearly perfectly (low spread). If the horizontal distance between the left shoulder and right shoulder is below 0.12 (in normalized 0-1 image coordinates), the app considers the user to be positioned sideways and proceeds with posture analysis.

### Skeleton connections

```python
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),  # Face
    (0, 4), (4, 5), (5, 6), (6, 8),  # Face
    (9, 10),                           # Mouth
    (11, 12),                          # Shoulders
    (11, 13), (13, 15),               # Left arm
    (12, 14), (14, 16),               # Right arm
    (11, 23), (12, 24),               # Torso
    (23, 24),                          # Hips
    (23, 25), (25, 27),               # Left leg
    (24, 26), (26, 28),               # Right leg
]
```

Each tuple is a pair of landmark indices that should be connected with a line when drawing the skeleton overlay. This maps out the human body — face, arms, torso, and legs — so the user can see a stick-figure representation of what MediaPipe detected.

## Functions

### `load_sounds(sounds_dir)`

```python
def load_sounds(sounds_dir):
    return sorted(glob(os.path.join(sounds_dir, "*.mp3")))
```

Takes a directory path, finds all `.mp3` files in it using `glob`, and returns them sorted alphabetically. The sorting ensures a consistent playback order (1.mp3, 2.mp3, 3.mp3, ...) rather than random filesystem ordering.

### `play_next_sound(sound_files)`

```python
def play_next_sound(sound_files):
    global audio_process, sound_index
    if not sound_files:
        return
    if audio_process and audio_process.poll() is None:
        return
    sound = sound_files[sound_index % len(sound_files)]
    sound_index += 1
    audio_process = subprocess.Popen(["afplay", sound])
```

Plays the next MP3 in sequence each time it's called. Uses two global variables to track state across calls:

- `audio_process` — the currently running `afplay` subprocess (or `None`). If `.poll()` returns `None`, audio is still playing, and the function returns early to avoid overlapping sounds.
- `sound_index` — an ever-incrementing counter. The modulo operator (`% len(sound_files)`) wraps it around so the sounds cycle endlessly: 1, 2, 3, ... 10, 1, 2, 3, ...

The `Popen` call launches `afplay` without blocking, so the webcam loop keeps running.

### `analyze_posture(landmarks)`

```python
def analyze_posture(landmarks):
    ear = landmarks[EAR]
    shoulder = landmarks[SHOULDER]
    hip = landmarks[HIP]

    neck_inclination = np.degrees(np.arctan2(
        ear.x - shoulder.x,
        shoulder.y - ear.y
    ))
    neck_angle = abs(neck_inclination)

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
```

This is the core posture analysis logic. It measures two angles:

1. **Neck angle** — the angle between the ear and shoulder, measured from vertical. `arctan2(horizontal_offset, vertical_distance)` gives 0 degrees when the ear is directly above the shoulder (perfect posture) and increases as the head moves forward. In MediaPipe's coordinate system, y increases downward, so `shoulder.y - ear.y` gives positive vertical distance.

2. **Torso angle** — the same calculation but between the shoulder and hip. This detects slouching or rounding of the upper back.

Both angles are taken as absolute values (it doesn't matter if you lean forward or backward — any deviation is bad). If either exceeds its threshold, posture is flagged as bad and a human-readable alert string is generated.

### `draw_landmarks(frame, landmarks)`

```python
def draw_landmarks(frame, landmarks):
    h, w = frame.shape[:2]

    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            if start.visibility > 0.5 and end.visibility > 0.5:
                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))
                cv2.line(frame, start_point, end_point, (255, 0, 0), 2)

    for i, landmark in enumerate(landmarks):
        if landmark.visibility > 0.5:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            if i in [EAR, SHOULDER, HIP]:
                cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)
            else:
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
```

Draws the detected pose as a skeleton overlay on the video frame. Two passes:

1. **Connections** — iterates over `POSE_CONNECTIONS` and draws blue lines between connected landmarks. Only draws if both endpoints have visibility above 0.5 (MediaPipe's confidence that the landmark is actually visible). Landmark coordinates are normalized (0.0 to 1.0), so they're multiplied by frame width/height to get pixel positions.

2. **Points** — iterates over all landmarks and draws circles. The three key posture landmarks (ear, shoulder, hip) get larger yellow circles (radius 8) so they stand out. All other landmarks get smaller green circles (radius 4).

### `main()`

The main function orchestrates everything. Here's the flow:

#### 1. Model loading

```python
model_path = os.path.join(script_dir, "pose_landmarker_lite.task")
if not os.path.exists(model_path):
    # prints download instructions and exits
```

Resolves the model path relative to the script and checks it exists before proceeding. If missing, it prints the exact `curl` command to download it.

#### 2. MediaPipe initialization

```python
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
landmarker = PoseLandmarker.create_from_options(options)
```

Creates a `PoseLandmarker` using the Tasks API. Key settings:

- **VIDEO mode** — optimized for sequential frames from a video source (as opposed to single images or live streams with callbacks). Requires monotonically increasing timestamps.
- **0.5 confidence thresholds** — the model must be at least 50% confident a pose exists to detect it, and 50% confident to maintain tracking between frames.

#### 3. Webcam and window setup

```python
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture(0)
```

Creates a resizable OpenCV window and opens the default webcam (device index 0).

#### 4. Sound loading

```python
sound_files = load_sounds(sounds_dir)
```

Loads MP3 files from `app/sounds/`. The app currently ships with 10 sound files (`1.mp3` through `10.mp3`).

#### 5. The main frame loop

```python
while True:
    ret, frame = cap.read()
```

This is the heart of the application — an infinite loop that processes one webcam frame per iteration:

**Frame capture and conversion:**

```python
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
```

OpenCV captures frames in BGR color order (Blue-Green-Red), but MediaPipe expects standard RGB. `cvtColor` handles the conversion. The frame is then wrapped in a MediaPipe `Image` object.

**Pose detection:**

```python
timestamp_ms = int((time.time() - start_time) * 1000)
results = landmarker.detect_for_video(mp_image, timestamp_ms)
```

Computes a millisecond timestamp relative to the start time (VIDEO mode requires these to be monotonically increasing) and runs pose detection on the frame.

**Sideways check:**

```python
shoulder_spread = abs(landmarks[LEFT_SHOULDER].x - landmarks[SHOULDER].x)
is_sideways = key_visible and shoulder_spread < SHOULDER_SPREAD_THRESHOLD
```

Before analyzing posture, the app checks whether the user is positioned sideways to the camera. If the left and right shoulders are close together in the frame (small x-spread), the user is sideways. If they're far apart, the user is facing the camera and posture analysis wouldn't be meaningful — instead, a centered yellow "POSITION YOURSELF SIDEWAYS" message is displayed.

**Posture analysis and display:**

When sideways, the app calls `analyze_posture()` and displays:
- A large green "GOOD POSTURE" or red "BAD POSTURE" label
- Neck and torso angle readings in white
- Specific alert messages in red (e.g., "Forward head: 23 deg")

**Audio alert logic:**

```python
if not is_good:
    if bad_posture_start is None:
        bad_posture_start = time.time()
    elif (not alerted_this_streak
          and time.time() - bad_posture_start >= BAD_POSTURE_DELAY
          and time.time() - last_alert_time > ALERT_COOLDOWN):
        play_next_sound(sound_files)
        last_alert_time = time.time()
        alerted_this_streak = True
else:
    bad_posture_start = None
    alerted_this_streak = False
```

This implements a three-stage alert system:

1. **First detection** — when posture first goes bad, `bad_posture_start` records the timestamp. No alert yet.
2. **Sustained bad posture** — once bad posture has persisted for `BAD_POSTURE_DELAY` seconds (3s) and the alert cooldown has elapsed, a sound plays and `alerted_this_streak` is set to prevent repeated alerts during the same streak.
3. **Recovery** — when posture returns to good, both the timer and the streak flag reset, so the next bad posture event starts fresh.

**Exit conditions:**

```python
if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
    break
if cv2.waitKey(10) & 0xFF == ord("q"):
    break
```

The loop exits if the user closes the window (clicking the X button) or presses the `q` key. `waitKey(10)` waits 10 milliseconds for a keypress, which also controls the effective frame rate.

#### 6. Cleanup

```python
landmarker.close()
cap.release()
cv2.destroyAllWindows()
```

Releases all resources: closes the MediaPipe model, releases the webcam, and destroys the OpenCV window.

## How to Run

### Prerequisites

- Python 3.14+
- macOS (audio alerts use `afplay`)
- A webcam
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/berohlfs/face-detection.git
cd face-detection

# Download the pose landmarker model
curl -o app/pose_landmarker_lite.task -L https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task

# Install dependencies and run
cd app
uv run main.py
```

### Usage

1. Sit sideways to your camera so it can see your profile (ear, shoulder, hip alignment)
2. The app will show a skeleton overlay and posture status in real time
3. If you slouch or lean your head forward for more than 3 seconds, an audio alert will play
4. Press `q` or close the window to exit
