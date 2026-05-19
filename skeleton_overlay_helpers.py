import cv2
import numpy as np

def landmarks_to_pixels(landmark_container, frame):
    h, w = frame.shape[:2]
    src = getattr(landmark_container, "landmark", landmark_container)
    pts = []
    for lm in src:
        if hasattr(lm, "x") and hasattr(lm, "y"):
            pts.append((int(lm.x * w), int(lm.y * h)))
        elif isinstance(lm, (list, tuple)) and len(lm) >= 2:
            pts.append((int(lm[0] * w), int(lm[1] * h)))
        else:
            pts.append(None)
    return pts

def draw_pose_connections(frame, pixel_points, connections, color=(0,255,0), thickness=2):
    for a, b in connections:
        ai = a.value if hasattr(a, "value") else int(a)
        bi = b.value if hasattr(b, "value") else int(b)
        if ai < len(pixel_points) and bi < len(pixel_points):
            pa = pixel_points[ai]
            pb = pixel_points[bi]
            if pa is None or pb is None:
                continue
            cv2.line(frame, pa, pb, color, thickness)



def calculate_angle_2_points(p1, p2):
    if p1 is None or p2 is None:
        return None
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def calculate_angle_3_points(p1, p2, p3):
    if p1 is None or p2 is None or p3 is None:
        return None
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    dot_prod = np.dot(v1, v2)
    mag_v1 = np.linalg.norm(v1)
    mag_v2 = np.linalg.norm(v2)
    if mag_v1 == 0 or mag_v2 == 0:
        return None
    cos_angle = dot_prod / (mag_v1 * mag_v2)    # linear algebra formula
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    return angle_deg