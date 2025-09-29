"""
Face detector + tracker with:
 - ONNX (GPU) inference fallback to Caffe
 - threaded MJPEG reader
 - per-track Kalman + mask smoothing + sparse-LK mask warp for large frames
 - sepia mask rendering with confidence smoothing + hysteresis
 - head-pose estimation (solvePnP) using facemark 68-landmarks when available
 - HUD showing stream FPS, processing FPS and inference latency
 - UDP telemetry includes head pose per track
 - Visual overlay showing per-track yaw/pitch (and small yaw arrow)
"""
import cv2
import numpy as np
import os
import sys
import time
import threading
import urllib.request
from pathlib import Path
import requests
from collections import deque
import csv
from datetime import datetime, timezone
import socket
import json
import math

# ONNX runtime
try:
    import onnxruntime as ort
except Exception:
    ort = None

# ========== Settings ==========
VIDEO_SOURCE = "http://192.168.2.164:8080/video?x.mjpg"
SNAPSHOT_URL = "http://192.168.2.164:8080/shot.jpg"
MODEL_DIR = "models/face_detector"
PROTOTXT = os.path.join(MODEL_DIR, "deploy.prototxt")
CAFFEMODEL = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
ONNX_MODEL = os.path.join(MODEL_DIR, "retinaface.onnx")
LBF_MODEL = os.path.join(MODEL_DIR, "lbfmodel.yaml")
CONF_THRESHOLD = 0.5

FACE_MARGIN = 0.18
BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75
EDGE_KERNEL_SIZE = (7, 7)
MASK_BLUR_KERNEL = (15, 15)
MASK_HISTORY = 6
BBOX_EMA_ALPHA = 0.6
TRACK_DROP_FRAMES = 12
IOU_MATCH_THRESH = 0.25

KF_PROCESS_NOISE = 1e-2
KF_MEAS_NOISE = 1e-1

CONF_ALPHA = 0.92
MASK_ENABLE_THRESH = 0.70
MASK_DISABLE_THRESH = 0.55
MASK_DISABLE_HOLD = 6

# Flow / sparse-LK selection
FLOW_PYRAMID_SCALE = 0.5
FLOW_LEVELS = 2
FLOW_WINSIZE = 15
FLOW_ITER = 3
FLOW_POLY_N = 5
FLOW_POLY_SIGMA = 1.2
FLOW_FLAGS = 0

# switch to sparse-LK when frame area > this (pixels)
SPARSE_LK_FRAME_AREA_THRESH = 640 * 480  # use sparse-LK for frames larger than VGA area
SPARSE_LK_MAX_CORNERS = 100
SPARSE_LK_QUALITY = 0.01
SPARSE_LK_MIN_DIST = 7
SPARSE_LK_WIN_SIZE = (21, 21)
SPARSE_LK_MAX_LEVEL = 3

# UDP telemetry
UDP_ENABLED = True
UDP_HOST = "127.0.0.1"
UDP_PORT = 22345

LOG_ENABLED = True
LOG_PATH = "tracks_log.csv"

# HUD smoothing
HUD_ALPHA = 0.85

# 3D model points for head-pose (units arbitrary, consistent)
FACE_3D_MODEL = np.array([
    (0.0, 0.0, 0.0),          # nose tip
    (0.0, -330.0, -65.0),     # chin
    (-225.0, 170.0, -135.0),  # left eye left corner
    (225.0, 170.0, -135.0),   # right eye right corner
    (-150.0, -150.0, -125.0), # left mouth corner
    (150.0, -150.0, -125.0)   # right mouth corner
], dtype=np.float32)
LANDMARK_IDX = {
    'nose_tip': 30,
    'chin': 8,
    'left_eye_outer': 36,
    'right_eye_outer': 45,
    'mouth_left': 48,
    'mouth_right': 54
}
# ===================================

PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/3.4.0/samples/dnn/face_detector/deploy.prototxt"
CAFFEMODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
LBF_MODEL_URL = "https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml"

# Clear CSV at start
with open(LOG_PATH, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "timestamp_utc_iso", "frame_idx", "track_id",
        "x1", "y1", "x2", "y2",
        "centroid_x", "centroid_y",
        "width", "height", "confidence",
        "pose_yaw_deg", "pose_pitch_deg", "pose_roll_deg",
        "frozen_flag"
    ])
print("CSV cleared — fresh session.")

# UDP socket
udp_sock = None
if UDP_ENABLED:
    try:
        udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_sock.setblocking(False)
        print(f"UDP telemetry -> {UDP_HOST}:{UDP_PORT}")
    except Exception as e:
        udp_sock = None
        print("UDP socket failed:", e)

# Globals
tracks = {}
_next_track_id = 0
frozen = False
use_facemark = False
facemark = None
_prev_gray = None

# HUD stats
_stream_fps = 0.0
_proc_fps = 0.0
_avg_infer_ms = 0.0

# Detector handles
onnx_session = None
cv2_net = None
detector_model_type = None

# --- utilities ---
def _next_id():
    global _next_track_id
    tid = _next_track_id
    _next_track_id += 1
    return tid

# small numeric helper
def clamp(x, a, b):
    """Clamp x to [a, b]. Works with ints/floats."""
    try:
        return max(a, min(b, x))
    except Exception:
        # if types incompatible, try float conversion
        try:
            xf = float(x)
            return max(a, min(b, xf))
        except Exception:
            return a


def download_stream(url, dest_path):
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        print(f"Downloading {url} ...")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        print("Download done.")
        return True
    except Exception as e:
        print("Download error:", e)
        return False

def ensure_model_files():
    ok = True
    if not os.path.exists(PROTOTXT):
        ok = download_stream(PROTOTXT_URL, PROTOTXT) and ok
    if not os.path.exists(CAFFEMODEL):
        ok = download_stream(CAFFEMODEL_URL, CAFFEMODEL) and ok
    return ok

def download_lbf_if_needed():
    if not os.path.exists(LBF_MODEL):
        print("Downloading LBF model...")
        return download_stream(LBF_MODEL_URL, LBF_MODEL)
    return True

def try_load_facemark():
    global facemark, use_facemark
    if not hasattr(cv2, "face"):
        print("cv2.face not available; facemark disabled.")
        use_facemark = False
        return None
    try:
        if not download_lbf_if_needed():
            use_facemark = False
            return None
        facemark_local = cv2.face.createFacemarkLBF()
        facemark_local.loadModel(LBF_MODEL)
        facemark = facemark_local
        use_facemark = True
        print("Facemark LBF loaded.")
    except Exception as e:
        print("Facemark load failed:", e)
        use_facemark = False

# MJPEG threaded reader
class MJPEGStream:
    def __init__(self, url, timeout=10, reconnect_delay=1.0):
        self.url = url
        self.timeout = timeout
        self.reconnect_delay = reconnect_delay
        self._running = False
        self._thread = None
        self._frame = None
        self._lock = threading.Lock()
        self.session = None
        self._frames_count = 0
        self._fps = 0.0
        self._fps_ts = time.time()

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self._running:
            try:
                self.session = requests.Session()
                with self.session.get(self.url, stream=True, timeout=self.timeout) as resp:
                    if resp.status_code != 200:
                        time.sleep(self.reconnect_delay)
                        continue
                    bytes_buf = b''
                    for chunk in resp.iter_content(chunk_size=8192):
                        if not self._running:
                            break
                        if chunk:
                            bytes_buf += chunk
                            start = bytes_buf.find(b'\xff\xd8')
                            end = bytes_buf.find(b'\xff\xd9', start + 2) if start != -1 else -1
                            if start != -1 and end != -1:
                                jpg = bytes_buf[start:end+2]
                                bytes_buf = bytes_buf[end+2:]
                                arr = np.frombuffer(jpg, dtype=np.uint8)
                                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                                if img is not None:
                                    with self._lock:
                                        self._frame = img
                                    self._frames_count += 1
                                    now = time.time()
                                    if now - self._fps_ts >= 1.0:
                                        self._fps = self._frames_count / (now - self._fps_ts)
                                        self._frames_count = 0
                                        self._fps_ts = now
            except Exception:
                time.sleep(self.reconnect_delay)
            finally:
                try:
                    if self.session is not None:
                        self.session.close()
                        self.session = None
                except Exception:
                    pass

    def read(self):
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def stop(self):
        self._running = False
        try:
            if self.session is not None:
                self.session.close()
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def fps(self):
        return self._fps

# Kalman
def create_kalman_for_bbox(bbox):
    kf = cv2.KalmanFilter(8, 4)
    kf.transitionMatrix = np.eye(8, dtype=np.float32)
    for i in range(4):
        kf.transitionMatrix[i, i+4] = 1.0
    kf.measurementMatrix = np.zeros((4, 8), dtype=np.float32)
    kf.measurementMatrix[0, 0] = 1.0
    kf.measurementMatrix[1, 1] = 1.0
    kf.measurementMatrix[2, 2] = 1.0
    kf.measurementMatrix[3, 3] = 1.0
    kf.processNoiseCov = np.eye(8, dtype=np.float32) * KF_PROCESS_NOISE
    kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * KF_MEAS_NOISE
    kf.errorCovPost = np.eye(8, dtype=np.float32) * 1.0

    x1, y1, x2, y2 = bbox
    cx = float((x1 + x2) / 2.0)
    cy = float((y1 + y2) / 2.0)
    w = float(max(1.0, x2 - x1))
    h = float(max(1.0, y2 - y1))
    state = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32).reshape(-1, 1)
    kf.statePost = state.copy()
    return kf

# Mask + edges
def _auto_canny_thresholds(gray, sigma=0.33):
    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    if lower >= upper:
        lower = max(0, upper - 1)
    return lower, upper

def refine_face_mask(roi_bgr):
    h, w = roi_bgr.shape[:2]
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray_filtered = cv2.bilateralFilter(gray, BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE)
    lower, upper = _auto_canny_thresholds(gray_filtered, sigma=0.33)
    edges = cv2.Canny(gray_filtered, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, EDGE_KERNEL_SIZE)
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(edges_closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros((h, w), dtype=np.uint8)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > (w * h) * 0.01:
            hull = cv2.convexHull(largest)
            cv2.drawContours(mask, [hull], -1, 255, thickness=-1)
        else:
            cv2.ellipse(mask, (w//2, h//2), (w//2, h//2), 0, 0, 360, 255, -1)
    else:
        cv2.ellipse(mask, (w//2, h//2), (int(w*0.45), int(h*0.55)), 0, 0, 360, 255, -1)
    mask_blurred = cv2.GaussianBlur(mask, MASK_BLUR_KERNEL, 0)
    edge_overlay = cv2.dilate(edges_closed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations=1)
    return mask_blurred, edge_overlay

def mask_from_landmarks(landmarks, bbox):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return None, None
    pts = np.array(landmarks, dtype=np.int32)
    if pts.ndim == 3:
        pts = pts.reshape(-1, 2)
    if np.any(pts[:,0] > w) or np.any(pts[:,1] > h):
        pts = pts - np.array([x1, y1], dtype=np.int32)
    pts[:,0] = np.clip(pts[:,0], 0, max(0, w-1))
    pts[:,1] = np.clip(pts[:,1], 0, max(0, h-1))
    mask = np.zeros((h, w), dtype=np.uint8)
    try:
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 255)
    except Exception:
        cv2.ellipse(mask, (w//2, h//2), (int(w*0.45), int(h*0.55)), 0, 0, 360, 255, -1)
    mask_blurred = cv2.GaussianBlur(mask, MASK_BLUR_KERNEL, 0)
    edges = np.zeros_like(mask)
    try:
        cv2.polylines(edges, [hull], True, 255, 1)
    except Exception:
        pass
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations=1)
    return mask_blurred, edges

# Tracking helpers
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    areaA = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    areaB = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
    union = areaA + areaB - interArea
    if union <= 0:
        return 0.0
    return interArea / union

def match_detections_to_tracks(detections, current_tracks):
    mapping = [None] * len(detections)
    if not current_tracks:
        return mapping
    used_tracks = set()
    for d_idx, det in enumerate(detections):
        best_tid = None
        best_iou = 0.0
        for tid, tr in current_tracks.items():
            if tid in used_tracks:
                continue
            ref_box = tr.get('bbox_pred', tr.get('bbox_smoothed'))
            if ref_box is None:
                continue
            i = iou(det['box'], ref_box)
            if i > best_iou:
                best_iou = i
                best_tid = tid
        if best_iou >= IOU_MATCH_THRESH:
            mapping[d_idx] = best_tid
            used_tracks.add(best_tid)
    return mapping

def create_fullframe_mask(frame_h, frame_w, bbox, local_mask):
    x1, y1, x2, y2 = bbox
    fh, fw = frame_h, frame_w
    full = np.zeros((fh, fw), dtype=np.uint8)
    h_roi, w_roi = local_mask.shape[:2]
    x1c = max(0, x1); y1c = max(0, y1); x2c = min(fw, x2); y2c = min(fh, y2)
    if x2c <= x1c or y2c <= y1c:
        return full
    dest_w = x2c - x1c
    dest_h = y2c - y1c
    if (dest_h, dest_w) != (h_roi, w_roi):
        resized = cv2.resize(local_mask, (dest_w, dest_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = local_mask
    full[y1c:y2c, x1c:x2c] = resized
    return full

def update_or_create_track(track_id, det, frame_h, frame_w):
    """
    If track_id is None -> create new track for det and return new id.
    Otherwise update existing track (smooth bbox, append masks/edges, update KF/confidence).
    det must contain keys: 'box' (x1,y1,x2,y2), 'mask_local', 'edges_local', optionally 'confidence' and 'pose'.
    """
    global tracks
    # Ensure det contains mask_local/edges_local
    bbox = det.get('box')
    if bbox is None:
        return None

    # New track
    if track_id is None:
        tid = _next_id()
        x1, y1, x2, y2 = bbox
        track = {
            'bbox_smoothed': (int(x1), int(y1), int(x2), int(y2)),
            'masks': deque(maxlen=MASK_HISTORY),
            'edges': deque(maxlen=MASK_HISTORY),
            'last_seen': 0,
            'confidence': float(det.get('confidence', 1.0)),
            'conf_smoothed': float(det.get('confidence', 1.0)),
            'mask_enabled': False,
            'mask_hold': 0,
            'pose': det.get('pose', (0.0, 0.0, 0.0)),
        }
        # create kalman filter for this bbox if helper exists
        try:
            track['kf'] = create_kalman_for_bbox((int(x1), int(y1), int(x2), int(y2)))
        except Exception:
            track['kf'] = None

        # create full-frame masks and edges and append
        mask_local = det.get('mask_local')
        edges_local = det.get('edges_local')
        if mask_local is None:
            # fallback to a small filled ellipse in bbox
            w = max(1, int(x2 - x1)); h = max(1, int(y2 - y1))
            mask_local = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(mask_local, (w//2, h//2), (int(w*0.45), int(h*0.55)), 0, 0, 360, 255, -1)
        if edges_local is None:
            edges_local = np.zeros_like(mask_local)

        mask_full = create_fullframe_mask(frame_h, frame_w, (int(x1), int(y1), int(x2), int(y2)), mask_local)
        edges_full = create_fullframe_mask(frame_h, frame_w, (int(x1), int(y1), int(x2), int(y2)), edges_local)
        track['masks'].append(mask_full)
        track['edges'].append(edges_full)

        tracks[tid] = track
        return tid

    # Update existing track
    tr = tracks.get(track_id)
    if tr is None:
        # if track disappeared, create new instead
        return update_or_create_track(None, det, frame_h, frame_w)

    x1, y1, x2, y2 = bbox
    bx1, by1, bx2, by2 = tr.get('bbox_smoothed', (x1, y1, x2, y2))

    # Exponential moving average (EMA) smoothing of bbox
    nbx1 = int(BBOX_EMA_ALPHA * bx1 + (1.0 - BBOX_EMA_ALPHA) * x1)
    nby1 = int(BBOX_EMA_ALPHA * by1 + (1.0 - BBOX_EMA_ALPHA) * y1)
    nbx2 = int(BBOX_EMA_ALPHA * bx2 + (1.0 - BBOX_EMA_ALPHA) * x2)
    nby2 = int(BBOX_EMA_ALPHA * by2 + (1.0 - BBOX_EMA_ALPHA) * y2)
    tr['bbox_smoothed'] = (nbx1, nby1, nbx2, nby2)

    # Append masks/edges (convert to full-frame)
    mask_local = det.get('mask_local')
    edges_local = det.get('edges_local')
    if mask_local is None:
        w = max(1, int(x2 - x1)); h = max(1, int(y2 - y1))
        mask_local = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask_local, (w//2, h//2), (int(w*0.45), int(h*0.55)), 0, 0, 360, 255, -1)
    if edges_local is None:
        edges_local = np.zeros_like(mask_local)

    mask_full = create_fullframe_mask(frame_h, frame_w, tr['bbox_smoothed'], mask_local)
    edges_full = create_fullframe_mask(frame_h, frame_w, tr['bbox_smoothed'], edges_local)
    tr['masks'].append(mask_full)
    tr['edges'].append(edges_full)

    # Update last seen and confidence measurement
    tr['last_seen'] = 0
    tr['confidence'] = float(det.get('confidence', tr.get('confidence', 1.0)))

    # Confidence smoothing (EMA)
    prev_sm = float(tr.get('conf_smoothed', tr['confidence']))
    tr['conf_smoothed'] = CONF_ALPHA * prev_sm + (1.0 - CONF_ALPHA) * tr['confidence']

    # Hysteresis for enabling/disabling mask rendering
    if not tr.get('mask_enabled', False) and tr['conf_smoothed'] > MASK_ENABLE_THRESH:
        tr['mask_enabled'] = True
        tr['mask_hold'] = 0
    if tr.get('mask_enabled', False):
        if tr['conf_smoothed'] < MASK_DISABLE_THRESH:
            tr['mask_hold'] = tr.get('mask_hold', 0) + 1
            if tr['mask_hold'] > MASK_DISABLE_HOLD:
                tr['mask_enabled'] = False
                tr['mask_hold'] = 0
        else:
            tr['mask_hold'] = 0

    # Kalman filter correction with measurement (cx,cy,w,h)
    if 'kf' in tr and tr['kf'] is not None:
        try:
            cx = float((x1 + x2) / 2.0)
            cy = float((y1 + y2) / 2.0)
            w = float(max(1.0, x2 - x1))
            h = float(max(1.0, y2 - y1))
            meas = np.array([cx, cy, w, h], dtype=np.float32).reshape(4,1)
            tr['kf'].correct(meas)
        except Exception:
            pass

    # update pose if provided
    if 'pose' in det:
        tr['pose'] = det.get('pose', tr.get('pose', (0.0, 0.0, 0.0)))

    return track_id


# ----------------- Sparse-LK warp helper (new) -----------------
def sparse_lk_warp_prev_mask_to_current(prev_mask_full, prev_gray, cur_gray, bbox):
    """
    Warp prev_mask (full-frame) to current frame for the given bbox using sparse LK + affine fit.
    Returns a full-frame warped mask (uint8) the same size as prev_mask_full.
    If warp fails, returns None.
    """
    try:
        h_full, w_full = prev_mask_full.shape[:2]
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            return None
        # crop roi
        sx, sy, ex, ey = x1, y1, x2, y2
        sx = max(0, sx); sy = max(0, sy); ex = min(w_full, ex); ey = min(h_full, ey)
        roi_mask = prev_mask_full[sy:ey, sx:ex]
        if roi_mask.size == 0:
            return None
        roi_h, roi_w = roi_mask.shape[:2]
        # build mask for goodFeaturesToTrack (points only inside mask)
        mask = (roi_mask > 10).astype(np.uint8) * 255
        # if mask area tiny, expand to whole roi
        if cv2.countNonZero(mask) < 10:
            mask = None
        # select points within ROI (coords relative to ROI)
        prev_gray_roi = prev_gray[sy:ey, sx:ex]
        # cv2.goodFeaturesToTrack expects 8-bit single-channel
        try:
            corners = cv2.goodFeaturesToTrack(prev_gray_roi,
                                              maxCorners=SPARSE_LK_MAX_CORNERS,
                                              qualityLevel=SPARSE_LK_QUALITY,
                                              minDistance=SPARSE_LK_MIN_DIST,
                                              mask=mask)
        except Exception:
            corners = None
        if corners is None or len(corners) < 6:
            return None
        pts_prev = corners.reshape(-1,2)
        # convert to full-frame coords for LK
        pts_prev_full = pts_prev + np.array([sx, sy], dtype=np.float32)

        # calc LK to find new pts in current gray
        pts_prev_full = pts_prev_full.astype(np.float32)
        pts_next, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, pts_prev_full, None,
                                                     winSize=SPARSE_LK_WIN_SIZE,
                                                     maxLevel=SPARSE_LK_MAX_LEVEL,
                                                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        if pts_next is None:
            return None
        st = st.reshape(-1)
        good_prev = pts_prev_full[st==1]
        good_next = pts_next[st==1]
        if len(good_prev) < 6 or len(good_next) < 6:
            return None
        # compute affine transform from good_prev to good_next (full-frame coords)
        M, inliers = cv2.estimateAffinePartial2D(good_prev, good_next, method=cv2.RANSAC, ransacReprojThreshold=4.0, maxIters=100)
        if M is None:
            return None
        # warp ROI mask with affine: need ROI-local transform -> adjust translation
        # create a 2x3 matrix that maps ROI coords to new ROI coords.
        # current M maps full-frame coordinates: x' = M * [x; y; 1]
        # for ROI-local warp, we compute M_local that maps (x - sx, y - sy) -> (x' - sx, y' - sy)
        # i.e., M_local = T(-[sx,sy]) * M * T([sx,sy])
        T1 = np.array([[1,0,-sx],[0,1,-sy],[0,0,1]], dtype=np.float32)
        T2 = np.array([[1,0,sx],[0,1,sy],[0,0,1]], dtype=np.float32)
        M3 = np.eye(3, dtype=np.float32)
        M_aff = np.vstack([M, [0,0,1]]).astype(np.float32)
        M_local = (T1 @ M_aff @ T2)
        M_local2x3 = M_local[:2, :]
        # warp roi_mask
        warped_roi = cv2.warpAffine(roi_mask, M_local2x3, (roi_w, roi_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # paste into full-frame mask
        full = np.zeros((h_full, w_full), dtype=np.uint8)
        full[sy:ey, sx:ex] = warped_roi
        # slight blur to smooth
        full = cv2.GaussianBlur(full, (9,9), 0)
        return full
    except Exception:
        return None

# Dense Farneback warp (kept as fallback)
def _warp_mask_by_flow(prev_mask, flow):
    h, w = prev_mask.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x - flow[..., 0]).astype(np.float32)
    map_y = (grid_y - flow[..., 1]).astype(np.float32)
    warped = cv2.remap(prev_mask, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped

# Sepia and HUD helpers
SEPIA_KERNEL_BGR = np.array([
    [0.131, 0.534, 0.272],
    [0.168, 0.686, 0.349],
    [0.189, 0.769, 0.393],
], dtype=np.float32)
def apply_sepia_to_frame(frame):
    transformed = cv2.transform(frame.astype(np.float32), SEPIA_KERNEL_BGR)
    transformed = np.clip(transformed, 0, 255).astype(np.uint8)
    return transformed

def draw_hud(img, stream_fps, proc_fps, infer_ms, num_tracks):
    txts = [
        f"stream_fps: {stream_fps:.1f}",
        f"proc_fps: {proc_fps:.1f}",
        f"infer: {infer_ms:.1f} ms",
        f"tracks: {num_tracks}"
    ]
    x = 8
    y = 20
    for t in txts:
        cv2.putText(img, t, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
        y += 18

# ONNX helpers
def init_onnx_runtime(onnx_path):
    if ort is None:
        return None
    if not os.path.exists(onnx_path):
        return None
    try:
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            sess = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider","CPUExecutionProvider"])
        else:
            sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        return sess
    except Exception as e:
        print("ONNX init failed:", e)
        return None

def load_detector_with_onnx_fallback():
    onnx_sess = None
    cv2_net_local = None
    model_type_local = None
    if ort is not None and os.path.exists(ONNX_MODEL):
        onnx_sess = init_onnx_runtime(ONNX_MODEL)
        if onnx_sess is not None:
            return onnx_sess, None, "onnx_runtime"
    if os.path.exists(PROTOTXT) and os.path.exists(CAFFEMODEL):
        try:
            cv2_net_local = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)
            model_type_local = "caffe"
            return None, cv2_net_local, model_type_local
        except Exception as e:
            print("Caffe load failed:", e)
    return None, None, None

def run_onnx_inference(sess, frame, target_input_shape=(300,300), mean=(104.0, 177.0, 123.0)):
    if sess is None:
        return None
    input_meta = sess.get_inputs()[0]
    in_shape = input_meta.shape
    ih, iw = target_input_shape
    use_nchw = True
    try:
        if len(in_shape) == 4:
            if in_shape[1] == 3 or in_shape[1] == '3':
                use_nchw = True
                ih = in_shape[2] if isinstance(in_shape[2], int) and in_shape[2] > 0 else ih
                iw = in_shape[3] if isinstance(in_shape[3], int) and in_shape[3] > 0 else iw
            elif in_shape[3] == 3 or in_shape[3] == '3':
                use_nchw = False
                ih = in_shape[1] if isinstance(in_shape[1], int) and in_shape[1] > 0 else ih
                iw = in_shape[2] if isinstance(in_shape[2], int) and in_shape[2] > 0 else iw
    except Exception:
        pass
    img = cv2.resize(frame, (int(iw), int(ih))).astype(np.float32)
    img = img - np.array(mean, dtype=np.float32)
    if use_nchw:
        inp = img.transpose(2,0,1)[None, :, :, :].astype(np.float32)
    else:
        inp = img[None, :, :, :].astype(np.float32)
    input_name = sess.get_inputs()[0].name
    try:
        outputs = sess.run(None, {input_name: inp})
        return outputs
    except Exception as e:
        print("ONNX infer failed:", e)
        return None

def parse_onnx_outputs(outputs, frame_w, frame_h):
    dets = []
    if outputs is None:
        return dets
    for out in outputs:
        if isinstance(out, np.ndarray) and out.ndim == 4 and out.shape[1] == 1 and out.shape[3] >= 7:
            arr = out
            for i in range(arr.shape[2]):
                confidence = float(arr[0,0,i,2])
                if confidence < CONF_THRESHOLD:
                    continue
                box = arr[0,0,i,3:7] * np.array([frame_w, frame_h, frame_w, frame_h])
                sx, sy, ex, ey = box.astype(int)
                dets.append({'box':(int(sx),int(sy),int(ex),int(ey)), 'confidence':confidence})
            if dets:
                return dets
    for out in outputs:
        if isinstance(out, np.ndarray) and out.ndim == 2 and out.shape[1] >= 5:
            arr = out
            for row in arr:
                score = float(row[4])
                if score < CONF_THRESHOLD:
                    continue
                x1 = row[0]; y1 = row[1]; x2 = row[2]; y2 = row[3]
                if 0.0 <= x1 <= 1.0 and 0.0 <= x2 <= 1.0 and 0.0 <= y1 <= 1.0 and 0.0 <= y2 <= 1.0:
                    sx = int(x1 * frame_w)
                    sy = int(y1 * frame_h)
                    ex = int(x2 * frame_w)
                    ey = int(y2 * frame_h)
                else:
                    sx = int(x1); sy = int(y1); ex = int(x2); ey = int(y2)
                dets.append({'box':(sx, sy, ex, ey), 'confidence':score})
            if dets:
                return dets
    boxes = None; scores = None
    for out in outputs:
        if isinstance(out, np.ndarray):
            if out.ndim == 2 and out.shape[1] == 4 and boxes is None:
                boxes = out
            elif out.ndim == 1 and scores is None:
                scores = out
            elif out.ndim == 2 and out.shape[1] == 1 and scores is None:
                scores = out.reshape(-1)
    if boxes is not None and scores is not None:
        N = min(boxes.shape[0], scores.shape[0])
        for i in range(N):
            score = float(scores[i])
            if score < CONF_THRESHOLD:
                continue
            x1, y1, x2, y2 = boxes[i]
            if 0.0 <= x1 <= 1.0 and 0.0 <= x2 <= 1.0 and 0.0 <= y1 <= 1.0 and 0.0 <= y2 <= 1.0:
                sx = int(x1 * frame_w)
                sy = int(y1 * frame_h)
                ex = int(x2 * frame_w)
                ey = int(y2 * frame_h)
            else:
                sx = int(x1); sy = int(y1); ex = int(x2); ey = int(y2)
            dets.append({'box':(sx, sy, ex, ey), 'confidence':score})
        if dets:
            return dets
    return dets

# Pose helpers
def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return (math.degrees(y), math.degrees(x), math.degrees(z))

def estimate_head_pose_from_landmarks(landmarks, frame_shape, camera_matrix=None, dist_coeffs=None):
    try:
        pts = np.asarray(landmarks).reshape(-1, 2)
        idx = LANDMARK_IDX
        image_points = np.array([
            pts[idx['nose_tip']],
            pts[idx['chin']],
            pts[idx['left_eye_outer']],
            pts[idx['right_eye_outer']],
            pts[idx['mouth_left']],
            pts[idx['mouth_right']],
        ], dtype=np.float32)
        h, w = frame_shape[:2]
        if camera_matrix is None:
            focal_length = w
            center = (w / 2.0, h / 2.0)
            camera_matrix = np.array([[focal_length, 0, center[0]],
                                       [0, focal_length, center[1]],
                                       [0, 0, 1]], dtype=np.float32)
        if dist_coeffs is None:
            dist_coeffs = np.zeros((4,1), dtype=np.float32)
        retval, rvec, tvec = cv2.solvePnP(FACE_3D_MODEL, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not retval:
            return (0.0, 0.0, 0.0)
        R, _ = cv2.Rodrigues(rvec)
        yaw_deg, pitch_deg, roll_deg = rotationMatrixToEulerAngles(R)
        return (yaw_deg, pitch_deg, roll_deg)
    except Exception:
        return (0.0, 0.0, 0.0)

def garbage_collect_tracks():
    """
    Increment last_seen for all tracks, apply light confidence decay and
    mask-disable logic, then remove tracks that have been missing for
    more than TRACK_DROP_FRAMES frames.
    """
    to_delete = []
    for tid, tr in list(tracks.items()):
        # ensure last_seen present
        tr['last_seen'] = tr.get('last_seen', 0) + 1

        # gentle confidence decay so long-lived tracks slowly drop if not updated
        if 'conf_smoothed' in tr:
            # decay factor (slightly <1)
            tr['conf_smoothed'] = float(tr['conf_smoothed']) * 0.98

            # if mask was enabled but confidence dropped well below threshold, disable it
            if tr.get('mask_enabled', False) and tr['conf_smoothed'] < (MASK_DISABLE_THRESH * 0.8):
                tr['mask_enabled'] = False
                tr['mask_hold'] = 0

        # if a track hasn't been seen for too many frames, mark for deletion
        if tr['last_seen'] > TRACK_DROP_FRAMES:
            to_delete.append(tid)

    # remove stale tracks
    for tid in to_delete:
        try:
            # free any heavy structures (Kalman filter) if present
            if tid in tracks and 'kf' in tracks[tid]:
                try:
                    # no explicit close for cv2.KalmanFilter but drop ref
                    tracks[tid]['kf'] = None
                except Exception:
                    pass
            del tracks[tid]
        except KeyError:
            pass


# Main processing (modified to use sparse-LK when appropriate and draw overlay)
def process_and_draw(frame, frozen_flag=False, log_writer=None, log_file=None, frame_idx=None, cam_matrix=None, dist_coeffs=None):
    global _prev_gray, _stream_fps, _proc_fps, _avg_infer_ms
    frame_h, frame_w = frame.shape[:2]
    detections = []

    start_t = time.time()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    use_sparse_lk = (frame_w * frame_h) >= SPARSE_LK_FRAME_AREA_THRESH

    # If sparse-LK: per-track sparse warp, else dense flow for all tracks
    if use_sparse_lk and _prev_gray is not None and _prev_gray.shape == gray.shape:
        # per-track sparse LK warp
        for tid, tr in list(tracks.items()):
            if len(tr['masks']) == 0:
                continue
            try:
                prev_mask = tr['masks'][-1]
                warped_full = sparse_lk_warp_prev_mask_to_current(prev_mask, _prev_gray, gray, tr.get('bbox_smoothed', (0,0,0,0)))
                if warped_full is not None:
                    tr['masks'].append(warped_full)
                else:
                    # fallback: keep previous mask (no warp) to avoid sudden disappearance
                    tr['masks'].append(prev_mask.copy())
                if len(tr['edges'])>0:
                    prev_edges = tr['edges'][-1]
                    warped_edges = sparse_lk_warp_prev_mask_to_current(prev_edges, _prev_gray, gray, tr.get('bbox_smoothed', (0,0,0,0)))
                    if warped_edges is not None:
                        tr['edges'].append(warped_edges)
                    else:
                        tr['edges'].append(prev_edges.copy())
            except Exception:
                pass
    else:
        # dense flow path
        flow = None
        if _prev_gray is not None and _prev_gray.shape == gray.shape:
            try:
                flow = cv2.calcOpticalFlowFarneback(_prev_gray, gray, None,
                                                    pyr_scale=FLOW_PYRAMID_SCALE,
                                                    levels=FLOW_LEVELS,
                                                    winsize=FLOW_WINSIZE,
                                                    iterations=FLOW_ITER,
                                                    poly_n=FLOW_POLY_N,
                                                    poly_sigma=FLOW_POLY_SIGMA,
                                                    flags=FLOW_FLAGS)
            except Exception:
                flow = None
        if flow is not None:
            for tid, tr in list(tracks.items()):
                try:
                    if len(tr['masks']) > 0:
                        prev_mask = tr['masks'][-1]
                        warped_mask = _warp_mask_by_flow(prev_mask, flow)
                        tr['masks'].append(warped_mask)
                    if len(tr['edges']) > 0:
                        prev_edges = tr['edges'][-1]
                        warped_edges = _warp_mask_by_flow(prev_edges, flow)
                        tr['edges'].append(warped_edges)
                except Exception:
                    pass

    # Kalman predict
    for tid, tr in list(tracks.items()):
        if 'kf' in tr and tr['kf'] is not None:
            try:
                pred = tr['kf'].predict()
                pcx = float(pred[0,0]); pcy = float(pred[1,0])
                pw = float(max(1.0, pred[2,0])); ph = float(max(1.0, pred[3,0]))
                px1 = int(pcx - pw/2.0); py1 = int(pcy - ph/2.0)
                px2 = int(pcx + pw/2.0); py2 = int(pcy + ph/2.0)
                px1 = max(0, min(frame_w-1, px1)); py1 = max(0, min(frame_h-1, py1))
                px2 = max(0, min(frame_w-1, px2)); py2 = max(0, min(frame_h-1, py2))
                tr['bbox_pred'] = (px1, py1, px2, py2)
            except Exception:
                tr['bbox_pred'] = tr.get('bbox_smoothed')

    infer_ms = 0.0
    # detection
    if not frozen_flag:
        if detector_model_type == "onnx_runtime" and onnx_session is not None:
            t0 = time.time()
            outputs = run_onnx_inference(onnx_session, frame, target_input_shape=(300,300))
            infer_ms = (time.time() - t0) * 1000.0
            parsed = parse_onnx_outputs(outputs, frame_w, frame_h)
            for pd in parsed:
                sx, sy, ex, ey = pd['box']
                bw = ex - sx; bh = ey - sy
                mx = int(bw * FACE_MARGIN); my = int(bh * FACE_MARGIN)
                startX = max(0, sx - mx); startY = max(0, sy - my)
                endX = min(frame_w - 1, ex + mx); endY = min(frame_h - 1, ey + my)
                if endX - startX < 10 or endY - startY < 10:
                    continue
                roi = frame[startY:endY, startX:endX].copy()
                pose_yaw = pose_pitch = pose_roll = 0.0
                if use_facemark and facemark is not None:
                    try:
                        rect = np.array([[startX, startY, endX - startX, endY - startY]], dtype=np.int32)
                        ok, shapes = facemark.fit(gray, rect)
                        if ok and shapes and len(shapes)>0:
                            lm = shapes[0].reshape(-1,2)
                            mask_local, edges_local = mask_from_landmarks(lm, (startX,startY,endX,endY))
                            if mask_local is None:
                                mask_local, edges_local = refine_face_mask(roi)
                            pose_yaw, pose_pitch, pose_roll = estimate_head_pose_from_landmarks(lm, frame.shape, camera_matrix=cam_matrix, dist_coeffs=dist_coeffs)
                        else:
                            mask_local, edges_local = refine_face_mask(roi)
                    except Exception:
                        mask_local, edges_local = refine_face_mask(roi)
                else:
                    mask_local, edges_local = refine_face_mask(roi)
                detections.append({
                    'box': (startX, startY, endX, endY),
                    'confidence': pd.get('confidence', 1.0),
                    'mask_local': mask_local,
                    'edges_local': edges_local,
                    'pose': (pose_yaw, pose_pitch, pose_roll)
                })
        elif detector_model_type == "caffe" and cv2_net is not None:
            t0 = time.time()
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (300,300), (104.0,177.0,123.0), swapRB=False, crop=False)
            cv2_net.setInput(blob)
            try:
                dets = cv2_net.forward()
            except Exception:
                dets = None
            infer_ms = (time.time() - t0) * 1000.0
            if dets is not None:
                for i in range(0, dets.shape[2]):
                    confidence = float(dets[0,0,i,2])
                    if confidence < CONF_THRESHOLD:
                        continue
                    box = dets[0,0,i,3:7] * np.array([frame_w, frame_h, frame_w, frame_h])
                    sx, sy, ex, ey = box.astype(int)
                    bw = ex - sx; bh = ey - sy
                    mx = int(bw * FACE_MARGIN); my = int(bh * FACE_MARGIN)
                    startX = max(0, sx - mx); startY = max(0, sy - my)
                    endX = min(frame_w - 1, ex + mx); endY = min(frame_h - 1, ey + my)
                    if endX - startX < 10 or endY - startY < 10:
                        continue
                    roi = frame[startY:endY, startX:endX].copy()
                    pose_yaw = pose_pitch = pose_roll = 0.0
                    if use_facemark and facemark is not None:
                        try:
                            rect = np.array([[startX, startY, endX - startX, endY - startY]], dtype=np.int32)
                            ok, shapes = facemark.fit(gray, rect)
                            if ok and shapes and len(shapes)>0:
                                lm = shapes[0].reshape(-1,2)
                                mask_local, edges_local = mask_from_landmarks(lm, (startX,startY,endX,endY))
                                if mask_local is None:
                                    mask_local, edges_local = refine_face_mask(roi)
                                pose_yaw, pose_pitch, pose_roll = estimate_head_pose_from_landmarks(lm, frame.shape, camera_matrix=cam_matrix, dist_coeffs=dist_coeffs)
                            else:
                                mask_local, edges_local = refine_face_mask(roi)
                        except Exception:
                            mask_local, edges_local = refine_face_mask(roi)
                    else:
                        mask_local, edges_local = refine_face_mask(roi)
                    detections.append({
                        'box': (startX, startY, endX, endY),
                        'confidence': confidence,
                        'mask_local': mask_local,
                        'edges_local': edges_local,
                        'pose': (pose_yaw, pose_pitch, pose_roll)
                    })
        else:
            pass

        mapping = match_detections_to_tracks(detections, tracks)
        used_track_ids = set()
        for d_idx, det in enumerate(detections):
            tid = mapping[d_idx]
            if tid is None:
                new_tid = update_or_create_track(None, det, frame_h, frame_w)
                tracks[new_tid]['pose'] = det.get('pose', (0.0,0.0,0.0))
                used_track_ids.add(new_tid)
            else:
                update_or_create_track(tid, det, frame_h, frame_w)
                tracks[tid]['pose'] = det.get('pose', (0.0,0.0,0.0))
                used_track_ids.add(tid)
        for tid in list(tracks.keys()):
            if tid not in used_track_ids:
                tracks[tid]['last_seen'] += 1

    # render sepia masked output and HUD
    output = frame.copy().astype(np.uint8)
    sepia_frame = apply_sepia_to_frame(frame)

    ts = datetime.now(timezone.utc).isoformat()
    if LOG_ENABLED and (frame_idx is not None):
        track_list_for_udp = []
        for tid, tr in list(tracks.items()):
            sx, sy, ex, ey = tr['bbox_smoothed']
            sx_i, sy_i, ex_i, ey_i = int(sx), int(sy), int(ex), int(ey)
            w = max(0, ex_i - sx_i)
            h = max(0, ey_i - sy_i)
            cx = sx_i + w//2
            cy = sy_i + h//2
            conf = tr.get('confidence', 0.0)
            yaw_p, pitch_p, roll_p = tr.get('pose', (0.0,0.0,0.0))
            # CSV log row (append)
            try:
                with open(LOG_PATH, "a", newline="") as lf:
                    lw = csv.writer(lf)
                    lw.writerow([ts, frame_idx, tid, sx_i, sy_i, ex_i, ey_i, cx, cy, w, h, float(conf), float(yaw_p), float(pitch_p), float(roll_p), int(frozen_flag)])
            except Exception:
                pass
            track_list_for_udp.append({
                "id": int(tid),
                "cx": int(cx),
                "cy": int(cy),
                "w": int(w),
                "h": int(h),
                "conf": float(conf),
                "pose_yaw_deg": float(yaw_p),
                "pose_pitch_deg": float(pitch_p),
                "pose_roll_deg": float(roll_p),
                "frozen": int(frozen_flag)
            })
        # send UDP
        if UDP_ENABLED and udp_sock is not None:
            payload = {"ts": ts, "frame_idx": int(frame_idx or -1), "tracks": track_list_for_udp}
            try:
                udp_sock.sendto(json.dumps(payload).encode('utf-8'), (UDP_HOST, UDP_PORT))
            except Exception:
                pass

    # rendering for each track with overlay (yaw/pitch) and small yaw arrow
    for tid, tr in list(tracks.items()):
        if len(tr['masks']) == 0:
            continue
        avg_mask = np.mean(np.stack(list(tr['masks']), axis=0), axis=0)
        alpha = (avg_mask / 255.0)[:, :, np.newaxis].astype(np.float32)
        avg_edges = np.mean(np.stack(list(tr['edges']), axis=0), axis=0)
        edges_bin = (avg_edges > 100).astype(np.uint8)

        if tr.get('mask_enabled', False):
            blended = (sepia_frame.astype(np.float32) * alpha + output.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
            edge_color = np.zeros_like(output)
            edge_color[edges_bin > 0] = (10, 255, 10)
            blended_with_edges = cv2.addWeighted(blended, 1.0, edge_color, 0.8, 0)
            mask_bool = (avg_mask > 10)
            mask3 = np.repeat(mask_bool[:, :, np.newaxis], 3, axis=2)
            output[mask3] = blended_with_edges[mask3]

        sx, sy, ex, ey = tr['bbox_smoothed']
        sx, sy, ex, ey = int(sx), int(sy), int(ex), int(ey)
        cv2.rectangle(output, (sx, sy), (ex, ey), (10, 255, 10), 1)

        # overlay: small filled rect above bbox with yaw/pitch text
        yaw_p, pitch_p, roll_p = tr.get('pose', (0.0,0.0,0.0))
        overlay_text = f"Yaw:{yaw_p:+.1f}°  Pitch:{pitch_p:+.1f}°"
        # calculate overlay pos (above bbox if space, else below)
        oy = sy - 18 if sy - 18 > 8 else ey + 12
        ox = sx
        # background rectangle
        (tw, th), _ = cv2.getTextSize(overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(output, (ox-2, oy-14), (ox + tw + 4, oy + 2), (0, 0, 0), thickness=-1)
        cv2.putText(output, overlay_text, (ox, oy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 220, 20), 1, cv2.LINE_AA)

        # small yaw arrow: from bbox center draw a short horizontal line left/right according to yaw sign
        cx = sx + (ex - sx)//2
        cy = sy + (ey - sy)//2
        # yaw->pixel mapping: assumes a small scale (5 px per degree) clipped to bbox width/4
        scale = 4.0
        dx = int(clamp(yaw_p * scale, - (ex - sx)//4, (ex - sx)//4))
        x2 = cx + dx
        y2 = cy
        cv2.arrowedLine(output, (cx, cy), (x2, y2), (200,200,50), 2, tipLength=0.3)

        label = f"id={tid} conf={tr.get('confidence',0):.2f}"
        ylab = sy - 28 if sy - 28 > 8 else sy + 28
        cv2.putText(output, label, (sx, ylab), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (10,255,10), 1)

    garbage_collect_tracks()

    # HUD: update and draw
    now = time.time()
    elapsed = now - start_t
    if elapsed > 0:
        frame_proc_fps = 1.0 / elapsed
        _proc_fps = HUD_ALPHA * _proc_fps + (1.0 - HUD_ALPHA) * frame_proc_fps if _proc_fps > 0 else frame_proc_fps
    _avg_infer_ms = HUD_ALPHA * _avg_infer_ms + (1.0 - HUD_ALPHA) * infer_ms if _avg_infer_ms > 0 else infer_ms
    num_tracks = len(tracks)
    draw_hud(output, _stream_fps, _proc_fps, _avg_infer_ms, num_tracks)

    _prev_gray = gray.copy()
    return output

def main():
    global frozen, onnx_session, cv2_net, detector_model_type, _stream_fps
    # model files
    if not (os.path.exists(PROTOTXT) and os.path.exists(CAFFEMODEL)):
        print("trying to download Caffe model files...")
        ensure_model_files()

    onnx_session, cv2_net, detector_model_type = load_detector_with_onnx_fallback()
    if detector_model_type is None:
        print("No detector available.")
        sys.exit(1)
    print("Using detector:", detector_model_type)
    if detector_model_type == "onnx_runtime" and ort is not None:
        print("ONNX providers:", ort.get_available_providers())

    try_load_facemark()

    print("Starting stream:", VIDEO_SOURCE)
    stream = MJPEGStream(VIDEO_SOURCE, timeout=10)
    stream.start()

    # allow a short warmup for stream
    first_wait = time.time()
    first_frame = None
    while time.time() - first_wait < 2.5:
        first_frame = stream.read()
        if first_frame is not None:
            break
        time.sleep(0.05)
    use_snapshot = False
    if first_frame is None:
        print("Stream empty, falling back to snapshot polling.")
        use_snapshot = True
    else:
        print("Stream OK.")
        use_snapshot = False

    frame_idx = 0
    cam_matrix = None
    dist_coeffs = None

    try:
        while True:
            if use_snapshot:
                frame = None
                try:
                    frame = grab_snapshot_frame(SNAPSHOT_URL)
                except Exception:
                    frame = None
                if frame is None:
                    maybe = stream.read()
                    if maybe is not None:
                        frame = maybe; use_snapshot = False
                    else:
                        time.sleep(0.02)
                        continue
            else:
                frame = stream.read()
                if frame is None:
                    print("Stream dropped; switching to snapshot.")
                    use_snapshot = True
                    continue

            _stream_fps = stream.fps() or _stream_fps

            t0 = time.time()
            out = process_and_draw(frame, frozen_flag=frozen, frame_idx=frame_idx, cam_matrix=cam_matrix, dist_coeffs=dist_coeffs)
            infer_latency = (time.time() - t0) * 1000.0

            frame_idx += 1
            title = "Face Detector (hud: stream/proc/infer) - f:freeze, c:clear, s:save, q:quit"
            if frozen:
                title += " [FROZEN]"
            cv2.imshow(title, out)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                fname = f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(fname, out)
                print("Saved", fname)
            elif key == ord("f"):
                frozen = not frozen
                print("Frozen." if frozen else "Unfrozen.")
            elif key == ord("c"):
                tracks.clear()
                print("Tracks cleared.")

    finally:
        try:
            stream.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        if udp_sock is not None:
            try:
                udp_sock.close()
            except Exception:
                pass

if __name__ == "__main__":
    main()


