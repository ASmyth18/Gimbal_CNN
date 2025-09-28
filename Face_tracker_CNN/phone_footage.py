""" face_detector.py (freeze/unfreeze + facial landmarks + temporal-stabilized mask
    + CSV track logging for gimbal control + sepia mono-colour)"""
import cv2
import numpy as np
import os
import sys
import time
import urllib.request
from pathlib import Path
import requests
from collections import deque
import csv
from datetime import datetime, timezone

# ========== User settings ==========
VIDEO_SOURCE = "http://192.168.2.164:8080/video?x.mjpg"   # preferred MJPEG endpoint
SNAPSHOT_URL = "http://192.168.2.164:8080/shot.jpg"       # fallback snapshot endpoint
MODEL_DIR = "models/face_detector"
PROTOTXT = os.path.join(MODEL_DIR, "deploy.prototxt")
CAFFEMODEL = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
LBF_MODEL = os.path.join(MODEL_DIR, "lbfmodel.yaml")   # facemark LBF model (optional)
CONF_THRESHOLD = 0.5

# Face mask/edge tuning (tweak these if needed)
FACE_MARGIN = 0.18            # expand bbox by this fraction to include hair/outline
BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75
EDGE_KERNEL_SIZE = (7, 7)     # morphological close kernel
MASK_BLUR_KERNEL = (15, 15)   # blur mask for smooth blend
MASK_HISTORY = 6              # number of past masks to average per track
BBOX_EMA_ALPHA = 0.6          # bbox smoothing factor (higher -> slower)
TRACK_DROP_FRAMES = 12        # how many frames to keep a track without detections
IOU_MATCH_THRESH = 0.25       # IoU threshold for matching detections to existing tracks

# CSV logging settings
LOG_ENABLED = True
LOG_PATH = "tracks_log.csv"
# ===================================

PROTOTXT_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/3.4.0/samples/dnn/face_detector/deploy.prototxt"
)
CAFFEMODEL_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
)
LBF_MODEL_URL = "https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml"


# --- Clear CSV at start of session ---
with open("tracks_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "timestamp_utc_iso", "frame_idx", "track_id",
        "x1", "y1", "x2", "y2",
        "centroid_x", "centroid_y",
        "width", "height", "confidence",
        "frozen_flag"
    ])
print("CSV log cleared — fresh session started.")


# --- global tracking & state ---
tracks = {}  # id -> track dict
_next_track_id = 0
frozen = False  # freeze/unfreeze toggle
use_facemark = False
facemark = None

# Sepia transform kernel for BGR (cv2.transform expects src BGR order)
SEPIA_KERNEL_BGR = np.array([
    [0.131, 0.534, 0.272],  # B' = 0.272*R + 0.534*G + 0.131*B  (reordered for BGR)
    [0.168, 0.686, 0.349],  # G'
    [0.189, 0.769, 0.393],  # R'
], dtype=np.float32)

def _next_id():
    global _next_track_id
    tid = _next_track_id
    _next_track_id += 1
    return tid

def download_stream(url, dest_path):
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        print(f"Downloading {url} ...")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(dest, "wb") as f:
                downloaded = 0
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded * 100 // total
                            print(f"\r  {pct}% ({downloaded}/{total} bytes)", end="", flush=True)
        if total:
            print("\r  100%")
        else:
            print("  Download complete")
        return True
    except Exception as e:
        print(f"\nDownload failed for {url}: {e}")
        if dest.exists():
            dest.unlink()
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
        print("LBF facemark model not found locally. Attempting to download...")
        return download_stream(LBF_MODEL_URL, LBF_MODEL)
    return True

def load_dnn_detector():
    try:
        net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)
        return net
    except Exception as e:
        print(f"Failed to load DNN model: {e}")
        return None

def load_haar_detector():
    haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if os.path.exists(haar_path):
        cascade = cv2.CascadeClassifier(haar_path)
        return cascade
    return None

def try_load_facemark():
    """
    Try to create facemark LBF if OpenCV has the face module and LBF file is present.
    Returns facemark object or None. Also sets global use_facemark accordingly.
    """
    global facemark, use_facemark
    if not hasattr(cv2, "face"):
        print("cv2.face module not available. Facial landmarks disabled.")
        use_facemark = False
        return None
    try:
        ok = download_lbf_if_needed()
        if not ok:
            print("Failed to download LBF model; facemark disabled.")
            use_facemark = False
            return None
        facemark_local = cv2.face.createFacemarkLBF()
        facemark_local.loadModel(LBF_MODEL)
        facemark = facemark_local
        use_facemark = True
        print("Facemark LBF loaded — using facial landmarks for masks.")
        return facemark
    except Exception as e:
        print(f"Facemark creation failed: {e}")
        use_facemark = False
        return None

def grab_snapshot_frame(url, timeout=5):
    """Return BGR frame from a JPEG snapshot URL or None on failure."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = resp.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    except Exception:
        return None

# ----------------- mask/edge refinement helpers -----------------
def _auto_canny_thresholds(gray, sigma=0.33):
    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    if lower >= upper:
        lower = max(0, upper - 1)
    return lower, upper

def refine_face_mask(roi_bgr):
    """
    Given a face ROI (BGR), return a soft mask (0..255) of the head region
    and an edge image. Uses bilateral filtering + auto-Canny + morphology + convex hull.
    """
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
    """
    landmarks: Nx2 array (absolute coordinates) OR coordinates relative to bbox.
    bbox: (x1,y1,x2,y2)
    returns local_mask (h_roi, w_roi) as uint8 and edge image (same size)
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return None, None
    pts = np.array(landmarks, dtype=np.int32)
    if pts.ndim == 3:  # sometimes returned as [[[x,y]], ...]
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
    # edge image from polygon boundary
    edges = np.zeros_like(mask)
    try:
        cv2.polylines(edges, [hull], True, 255, 1)
    except Exception:
        pass
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations=1)
    return mask_blurred, edges

# ----------------- tracking helpers -----------------
def iou(boxA, boxB):
    # boxes: (x1,y1,x2,y2)
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
            i = iou(det['box'], tr['bbox_smoothed'])
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
    if track_id is None:
        tid = _next_id()
        bbox = det['box']
        track = {
            'bbox_smoothed': bbox,
            'masks': deque(maxlen=MASK_HISTORY),
            'edges': deque(maxlen=MASK_HISTORY),
            'last_seen': 0,
            'confidence': det.get('confidence', 1.0),
        }
        mask_full = create_fullframe_mask(frame_h, frame_w, bbox, det['mask_local'])
        edges_full = create_fullframe_mask(frame_h, frame_w, bbox, det['edges_local'])
        track['masks'].append(mask_full)
        track['edges'].append(edges_full)
        tracks[tid] = track
        return tid
    else:
        tr = tracks[track_id]
        x1,y1,x2,y2 = det['box']
        bx1, by1, bx2, by2 = tr['bbox_smoothed']
        nbx1 = int(BBOX_EMA_ALPHA * bx1 + (1.0 - BBOX_EMA_ALPHA) * x1)
        nby1 = int(BBOX_EMA_ALPHA * by1 + (1.0 - BBOX_EMA_ALPHA) * y1)
        nbx2 = int(BBOX_EMA_ALPHA * bx2 + (1.0 - BBOX_EMA_ALPHA) * x2)
        nby2 = int(BBOX_EMA_ALPHA * by2 + (1.0 - BBOX_EMA_ALPHA) * y2)
        tr['bbox_smoothed'] = (nbx1, nby1, nbx2, nby2)
        mask_full = create_fullframe_mask(frame_h, frame_w, tr['bbox_smoothed'], det['mask_local'])
        edges_full = create_fullframe_mask(frame_h, frame_w, tr['bbox_smoothed'], det['edges_local'])
        tr['masks'].append(mask_full)
        tr['edges'].append(edges_full)
        tr['last_seen'] = 0
        tr['confidence'] = det.get('confidence', tr.get('confidence', 1.0))
        return track_id

def garbage_collect_tracks():
    to_delete = []
    for tid, tr in list(tracks.items()):
        tr['last_seen'] += 1
        if tr['last_seen'] > TRACK_DROP_FRAMES:
            to_delete.append(tid)
    for tid in to_delete:
        del tracks[tid]

# ----------------- main processing -----------------
def apply_sepia_to_frame(frame):
    """
    Apply sepia tint to a BGR frame using SEPIA_KERNEL_BGR via cv2.transform.
    Returns uint8 BGR image clipped to [0,255].
    """
    # cv2.transform expects float32
    transformed = cv2.transform(frame.astype(np.float32), SEPIA_KERNEL_BGR)
    transformed = np.clip(transformed, 0, 255).astype(np.uint8)
    return transformed

def process_and_draw(frame, net, cascade, use_haar, frozen_flag=False, log_writer=None, log_file=None, frame_idx=None):
    """
    Detect faces this frame, create local masks, match to tracks, update tracks,
    then apply averaged masks from tracks to the frame to produce sepia-tinted faces.
    Also logs track positions to CSV if log_writer provided.
    """
    frame_h, frame_w = frame.shape[:2]
    detections = []

    if not frozen_flag:
        if not use_haar:
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                         1.0, (300, 300),
                                         (104.0, 177.0, 123.0), swapRB=False, crop=False)
            net.setInput(blob)
            try:
                dets = net.forward()
            except Exception as e:
                print(f"DNN forward error: {e}")
                dets = None
            if dets is not None:
                for i in range(0, dets.shape[2]):
                    confidence = float(dets[0, 0, i, 2])
                    if confidence < CONF_THRESHOLD:
                        continue
                    box = dets[0, 0, i, 3:7] * np.array([frame_w, frame_h, frame_w, frame_h])
                    sx, sy, ex, ey = box.astype("int")
                    bw = ex - sx
                    bh = ey - sy
                    mx = int(bw * FACE_MARGIN)
                    my = int(bh * FACE_MARGIN)
                    startX = max(0, sx - mx)
                    startY = max(0, sy - my)
                    endX = min(frame_w - 1, ex + mx)
                    endY = min(frame_h - 1, ey + my)
                    if endX - startX < 10 or endY - startY < 10:
                        continue
                    roi = frame[startY:endY, startX:endX].copy()

                    # try landmarks first (if available)
                    if use_facemark and facemark is not None:
                        try:
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            rect = np.array([[startX, startY, endX - startX, endY - startY]], dtype=np.int32)
                            ok, shapes = facemark.fit(gray, rect)
                            if ok and shapes and len(shapes) > 0:
                                lm = shapes[0].reshape(-1, 2)
                                mask_local, edges_local = mask_from_landmarks(lm, (startX, startY, endX, endY))
                                if mask_local is None:
                                    mask_local, edges_local = refine_face_mask(roi)
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
                        'edges_local': edges_local
                    })
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                             minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
            for (x, y, fw, fh) in faces:
                mx = int(fw * FACE_MARGIN)
                my = int(fh * FACE_MARGIN)
                sx = max(0, x - mx)
                sy = max(0, y - my)
                ex = min(frame_w - 1, x + fw + mx)
                ey = min(frame_h - 1, y + fh + my)
                if ex - sx < 10 or ey - sy < 10:
                    continue
                roi = frame[sy:ey, sx:ex].copy()
                if use_facemark and facemark is not None:
                    try:
                        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        rect = np.array([[sx, sy, ex - sx, ey - sy]], dtype=np.int32)
                        ok, shapes = facemark.fit(gray_full, rect)
                        if ok and shapes and len(shapes) > 0:
                            lm = shapes[0].reshape(-1, 2)
                            mask_local, edges_local = mask_from_landmarks(lm, (sx, sy, ex, ey))
                            if mask_local is None:
                                mask_local, edges_local = refine_face_mask(roi)
                        else:
                            mask_local, edges_local = refine_face_mask(roi)
                    except Exception:
                        mask_local, edges_local = refine_face_mask(roi)
                else:
                    mask_local, edges_local = refine_face_mask(roi)

                detections.append({
                    'box': (sx, sy, ex, ey),
                    'confidence': 1.0,
                    'mask_local': mask_local,
                    'edges_local': edges_local
                })

        # match detections to existing tracks
        mapping = match_detections_to_tracks(detections, tracks)
        used_track_ids = set()
        for d_idx, det in enumerate(detections):
            tid = mapping[d_idx]
            if tid is None:
                new_tid = update_or_create_track(None, det, frame_h, frame_w)
                used_track_ids.add(new_tid)
            else:
                update_or_create_track(tid, det, frame_h, frame_w)
                used_track_ids.add(tid)
        # increment last_seen for unmatched tracks
        for tid in list(tracks.keys()):
            if tid not in used_track_ids:
                tracks[tid]['last_seen'] += 1

    # Now render output by applying averaged masks from tracks with sepia blending
    output = frame.copy().astype(np.uint8)

    # Precompute sepia version of the whole frame for blending
    sepia_frame = apply_sepia_to_frame(frame)

    # optional logging: write current track positions for gimbal control
    if LOG_ENABLED and log_writer is not None and frame_idx is not None:
        ts = datetime.now(timezone.utc).isoformat()
        for tid, tr in list(tracks.items()):
            sx, sy, ex, ey = tr['bbox_smoothed']
            sx_i, sy_i, ex_i, ey_i = int(sx), int(sy), int(ex), int(ey)
            w = max(0, ex_i - sx_i)
            h = max(0, ey_i - sy_i)
            cx = sx_i + w // 2
            cy = sy_i + h // 2
            conf = tr.get('confidence', 0.0)
            log_writer.writerow([
                ts, frame_idx, tid, sx_i, sy_i, ex_i, ey_i, cx, cy, w, h, float(conf), int(frozen_flag)
            ])
        try:
            # ensure it's flushed to disk for real-time gimbal control
            log_file.flush()
            os.fsync(log_file.fileno())
        except Exception:
            pass

    for tid, tr in list(tracks.items()):
        if len(tr['masks']) == 0:
            continue
        avg_mask = np.mean(np.stack(list(tr['masks']), axis=0), axis=0)
        alpha = (avg_mask / 255.0)[:, :, np.newaxis].astype(np.float32)
        avg_edges = np.mean(np.stack(list(tr['edges']), axis=0), axis=0)
        edges_bin = (avg_edges > 100).astype(np.uint8)

        # blend sepia_frame and output based on alpha
        blended = (sepia_frame.astype(np.float32) * alpha + output.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)

        # edge overlay (green-ish)
        edge_color = np.zeros_like(output)
        edge_color[edges_bin > 0] = (10, 255, 10)
        blended_with_edges = cv2.addWeighted(blended, 1.0, edge_color, 0.8, 0)

        mask_bool = (avg_mask > 10)
        mask3 = np.repeat(mask_bool[:, :, np.newaxis], 3, axis=2)
        output[mask3] = blended_with_edges[mask3]

        sx, sy, ex, ey = tr['bbox_smoothed']
        sx, sy, ex, ey = int(sx), int(sy), int(ex), int(ey)
        cv2.rectangle(output, (sx, sy), (ex, ey), (10, 255, 10), 1)
        label = f"face (id={tid}) {tr.get('confidence',0)*100:.0f}%"
        y = sy - 8 if sy - 8 > 8 else sy + 12
        cv2.putText(output, label, (sx, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 255, 10), 1)

    garbage_collect_tracks()
    return output

def main():
    global frozen
    # ensure DNN model files (attempt download if missing)
    dnn_ok = False
    if not (os.path.exists(PROTOTXT) and os.path.exists(CAFFEMODEL)):
        print("DNN model files not found locally. Attempting to download...")
        dnn_ok = ensure_model_files()
    else:
        dnn_ok = True

    net = None
    if dnn_ok:
        net = load_dnn_detector()

    use_haar = False
    cascade = None
    if net is None:
        print("Using Haar cascade fallback (not a CNN).")
        cascade = load_haar_detector()
        if cascade is None or cascade.empty():
            print("Error: Haar cascade not available. Please install OpenCV correctly.")
            sys.exit(1)
        use_haar = True
    else:
        print("Using DNN (SSD res10) face detector.")

    # try facemark
    try_load_facemark()

    # prepare CSV logging
    log_file = None
    log_writer = None
    if LOG_ENABLED:
        try:
            is_new = not os.path.exists(LOG_PATH)
            log_file = open(LOG_PATH, "a", newline="")
            log_writer = csv.writer(log_file)
            if is_new:
                log_writer.writerow([
                    "timestamp_utc_iso", "frame_idx", "track_id",
                    "x1", "y1", "x2", "y2", "centroid_x", "centroid_y",
                    "width", "height", "confidence", "frozen_flag"
                ])
                log_file.flush()
            print(f"Logging tracks to {LOG_PATH}")
        except Exception as e:
            print(f"Failed to open log file {LOG_PATH}: {e}")
            log_file = None
            log_writer = None

    print(f"Attempting to open stream: {VIDEO_SOURCE}")
    cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_FFMPEG)
    use_snapshot_fallback = False
    if not cap.isOpened():
        print("VideoCapture failed to open MJPEG stream. Falling back to snapshot polling.")
        use_snapshot_fallback = True

    frame_idx = 0
    try:
        while True:
            if use_snapshot_fallback:
                frame = grab_snapshot_frame(SNAPSHOT_URL)
                if frame is None:
                    time.sleep(0.1)
                    continue
            else:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("Stream dropped or no frame received — switching to snapshot fallback.")
                    use_snapshot_fallback = True
                    continue

            out = process_and_draw(frame, net, cascade, use_haar, frozen_flag=frozen,
                                   log_writer=log_writer, log_file=log_file, frame_idx=frame_idx)
            frame_idx += 1

            title = "Face Detector - Live (f:freeze, c:clear, s:save, q:quit)"
            if frozen:
                title += " [FROZEN]"
            cv2.imshow(title, out)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                fname = f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(fname, out)
                print(f"Saved {fname}")
            elif key == ord("f"):
                frozen = not frozen
                print("Frozen." if frozen else "Unfrozen, resuming detection.")
            elif key == ord("c"):
                tracks.clear()
                print("Cleared all tracks.")

    finally:
        if not use_snapshot_fallback:
            cap.release()
        cv2.destroyAllWindows()
        if log_file is not None:
            try:
                log_file.close()
            except Exception:
                pass

if __name__ == "__main__":
    main()
