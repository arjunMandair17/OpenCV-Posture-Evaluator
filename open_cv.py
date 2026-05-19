import os
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from skeleton_overlay_helpers import landmarks_to_pixels, draw_pose_connections, calculate_angle_2_points, calculate_angle_3_points

model_path = os.path.join(os.path.dirname(__file__), 'models', 'pose_landmarker_full.task')

base_options = python.BaseOptions(model_asset_path=model_path)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
)

detector = vision.PoseLandmarker.create_from_options(options)


import cv2

stream = cv2.VideoCapture(0)

# Lower camera resolution to reduce inference latency (minimal change)
stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Process every Nth frame to reduce CPU/GPU load
PROCESS_EVERY = 2
frame_counter = 0
last_result = None

badPostureFrameCount = 0
stage = "CALIBRATION" # Start in calibration stage to establish baseline metrics
calibrationFrameCount = 0

shoulder_angle_calibration_sum = 0
shoulder_nose_y_distance_norm_calibration_sum = 0
shoulder_nose_depth_diff_norm_calibration_sum = 0

avg_shoulder_angle = 0
avg_shoulder_nose_depth_diff_norm = 0
avg_shoulder_nose_y_distance_norm = 0

# Exponential moving average for jitter reduction
EMA_ALPHA = 0.2
smoothed_depth_ratio = None

POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),
    (0,4),(4,5),(5,6),(6,8),
    (9,10),
    (11,12),
    (11,13),(13,15),(15,17),(15,19),(15,21),
    (12,14),(14,16),(16,18),(16,20),(16,22),
    (11,23),(12,24),
    (23,24),
    (23,25),(25,27),(27,29),(29,31),
    (24,26),(26,28),(28,30),(30,32)
]


if not stream.isOpened():
    print("Cannot open camera")
    exit()

def euclidean_distance(p1, p2):
    if p1 is None or p2 is None:
        return None
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

while True:
    ret, frame = stream.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Only run inference on every Nth frame (or if we have no cached result)
    if frame_counter % PROCESS_EVERY == 0 or last_result is None:
        result = detector.detect_for_video(mp_image, frame_counter)
        last_result = result
    else:
        result = last_result

    frame_counter += 1

    if result and result.pose_landmarks:
        first = result.pose_landmarks[0]           # your LandmarkList-like object
        pts = landmarks_to_pixels(first, frame)
        draw_pose_connections(frame, pts, POSE_CONNECTIONS)

        landmarks = result.pose_landmarks[0]

        height, width, _ = frame.shape

        for landmark in landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)


        

        # Calculate angles and posture quality (normalized by shoulder width)
        shoulder_angle = abs(calculate_angle_2_points(pts[11], pts[12]))  # Left to right shoulder
        shoulder_nose_depth_diff = abs(landmarks[0].z - landmarks[11].z)  # Depth difference between nose and left shoulder
        shoulder_nose_y_distance = abs(pts[0][1] - pts[11][1])  # Vertical distance between nose and left shoulder

        shoulder_width = euclidean_distance(pts[11], pts[12])
        # Avoid dividing by tiny z-span values that amplify noise
        shoulder_depth_span = max(abs(landmarks[11].z - landmarks[12].z), 0.03)

        if shoulder_width is not None and shoulder_width > 1e-6 and shoulder_angle is not None:
            shoulder_nose_y_distance_norm = shoulder_nose_y_distance / shoulder_width
            shoulder_nose_depth_diff_norm = shoulder_nose_depth_diff / (shoulder_depth_span + 1e-6)

            # Smooth noisy metrics over time
            if smoothed_depth_ratio is None:
                smoothed_depth_ratio = shoulder_nose_depth_diff_norm
            else:
                smoothed_depth_ratio = EMA_ALPHA * shoulder_nose_depth_diff_norm + (1 - EMA_ALPHA) * smoothed_depth_ratio
            
            if stage == "CALIBRATION":
                cv2.putText(frame, 'Calibrating... Please maintain good posture', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                calibrationFrameCount += 1

                shoulder_angle_calibration_sum += shoulder_angle
                shoulder_nose_y_distance_norm_calibration_sum += shoulder_nose_y_distance_norm
                shoulder_nose_depth_diff_norm_calibration_sum += shoulder_nose_depth_diff_norm

                if calibrationFrameCount >= 300:
                    avg_shoulder_angle = shoulder_angle_calibration_sum / calibrationFrameCount
                    avg_shoulder_nose_y_distance_norm = shoulder_nose_y_distance_norm_calibration_sum / calibrationFrameCount
                    avg_shoulder_nose_depth_diff_norm = shoulder_nose_depth_diff_norm_calibration_sum / calibrationFrameCount

                    print("Calibration complete:")
                    print(f"Average Shoulder Angle: {avg_shoulder_angle:.2f} deg")
                    print(f"Average Nose-Shoulder Y Ratio: {avg_shoulder_nose_y_distance_norm:.2f}")
                    print(f"Average Nose-Shoulder Depth Ratio: {avg_shoulder_nose_depth_diff_norm:.2f}")

                    stage = "MONITORING"
                    
            elif stage == "MONITORING":

                # These ratios are camera-distance invariant; tune thresholds with real data.
                posture_quality = "Good" if abs(shoulder_angle - avg_shoulder_angle) < 10 and abs(shoulder_nose_y_distance_norm - avg_shoulder_nose_y_distance_norm) < 0.4 and abs(smoothed_depth_ratio - avg_shoulder_nose_depth_diff_norm) < 10 else "Bad"
                cv2.putText(frame, f'Shoulder Angle: {int(shoulder_angle)} deg', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f'Nose-Shoulder Depth Ratio: {smoothed_depth_ratio:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f'Nose-Shoulder Y Ratio: {shoulder_nose_y_distance_norm:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
                if posture_quality == "Good":
                    badPostureFrameCount = 0
                    cv2.putText(frame, f'Posture: {posture_quality}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, f'Posture: {posture_quality}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    badPostureFrameCount += 1
                    if badPostureFrameCount > 30:  # Alert if bad posture persists for a while
                        cv2.putText(frame, 'ALERT: Adjust Your Posture!', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)



    cv2.imshow('Pose Tracking', frame)
    if cv2.waitKey(1) == ord('q'):
        break

stream.release()
cv2.destroyAllWindows()