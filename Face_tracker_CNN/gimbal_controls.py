#!/usr/bin/env python3
"""
gimbal_tracker.py

Tails a CSV log created by the face detector (tracks_log.csv) and sends yaw/pitch
commands to a gimbal to steer it toward the selected face.

CSV expected columns (as produced by detector):
timestamp_utc_iso, frame_idx, track_id, x1, y1, x2, y2, centroid_x, centroid_y,
width, height, confidence, frozen_flag

Usage examples:
    # Print commands (test)
    python gimbal_tracker.py --csv tracks_log.csv --mode print --hfov 62 --vfov 44 --frame-w 640 --frame-h 480

    # Send via serial (e.g., COM3 on Windows)
    python gimbal_tracker.py --csv tracks_log.csv --mode serial --serial-port COM3 --baud 115200

    # Send via UDP
    python gimbal_tracker.py --csv tracks_log.csv --mode udp --udp-host 192.168.2.10 --udp-port 9000
"""

import argparse
import csv
import os
import time
import sys
import socket
import math
from datetime import datetime, timezone

try:
    import serial
except Exception:
    serial = None

# -------------------- Defaults / tuning --------------------
DEFAULT_CSV = "tracks_log.csv"
DEFAULT_HFOV_DEG = 62.0   # horizontal field of view of camera in degrees (tweak to camera)
DEFAULT_VFOV_DEG = 44.0   # vertical field of view in degrees
DEFAULT_FRAME_W = 640     # resolution used by detector (set to match detector)
DEFAULT_FRAME_H = 480

# PD controller gains (tweak)
KP_YAW = 0.9   # proportional gain for yaw (deg -> deg command)
KD_YAW = 0.15  # derivative gain for yaw
KP_PITCH = 0.9
KD_PITCH = 0.15

MAX_YAW_SPEED_DEG = 40.0    # limit on command magnitude (deg/s or deg step depending on your gimbal)
MAX_PITCH_SPEED_DEG = 40.0

STALE_TRACK_SECONDS = 1.0   # how old a track record can be before considered stale
TARGET_HOLD_TIME = 0.5      # keep target if updated within this time (seconds)

READ_LOOP_SLEEP = 0.02      # how long to sleep each tail loop (s)

# choose selection strategy when multiple tracks: 'largest' (by area) or 'highest_conf'
DEFAULT_SELECTION = "largest"

# -------------------- Helpers --------------------
def deg(x):
    return float(x)

def clamp(x, a, b):
    return max(a, min(b, x))

# Convert pixel offset into angle degrees
def pixel_offset_to_angle(dx_pixels, dy_pixels, frame_w, frame_h, hfov_deg, vfov_deg):
    # Normalized error in [-1,1] across half-width/height
    nx = dx_pixels / (frame_w / 2.0)
    ny = dy_pixels / (frame_h / 2.0)
    # Map to angular error using half-FOV
    yaw_deg = nx * (hfov_deg / 2.0)
    pitch_deg = ny * (vfov_deg / 2.0)
    return yaw_deg, pitch_deg

# -------------------- Output protocols --------------------
def send_command_print(yaw_cmd_deg, pitch_cmd_deg):
    # For debugging: simply print
    print(f"CMD -> YAW:{yaw_cmd_deg:+.3f}°, PITCH:{pitch_cmd_deg:+.3f}°")

def send_command_serial(ser, yaw_cmd_deg, pitch_cmd_deg):
    # Default ASCII format: "Y{yaw:.2f},P{pitch:.2f}\n"
    # Adapt to your gimbal's protocol here.
    msg = f"Y{yaw_cmd_deg:.2f},P{pitch_cmd_deg:.2f}\n"
    ser.write(msg.encode("ascii"))

def send_command_udp(sock, addr, yaw_cmd_deg, pitch_cmd_deg):
    # Send simple CSV line over UDP
    msg = f"{yaw_cmd_deg:.3f},{pitch_cmd_deg:.3f}"
    sock.sendto(msg.encode("ascii"), addr)

# -------------------- CSV tailer --------------------
def tail_csv_realtime(csv_path, on_new_rows, poll_sleep=READ_LOOP_SLEEP):
    """
    Open CSV file and yield newly appended rows as lists (parsed by csv.reader).
    on_new_rows is a callable that receives a list of parsed rows (one or more).
    This function will attempt to handle log rotation by re-opening if file inode changes.
    """
    last_size = 0
    with open(csv_path, "r", newline="") as f:
        # skip header if present
        reader = csv.reader(f)
        header = None
        try:
            header = next(reader)
        except StopIteration:
            header = None
        # Move to file end to only receive new rows
        f.seek(0, os.SEEK_END)
        buf = ""
        while True:
            where = f.tell()
            line = f.readline()
            if not line:
                # no new line yet
                time.sleep(poll_sleep)
                # handle file truncation/rotation
                try:
                    if os.path.getsize(csv_path) < where:
                        # file truncated/rotated - reopen
                        f.close()
                        f = open(csv_path, "r", newline="")
                        reader = csv.reader(f)
                        # skip header
                        try:
                            next(reader)
                        except Exception:
                            pass
                        f.seek(0, os.SEEK_END)
                except Exception:
                    pass
                continue
            # parse this line (strip possible newline)
            row = None
            try:
                # csv.reader expects an iterator - use csv.reader on [line]
                row = next(csv.reader([line]))
            except Exception:
                continue
            if not row:
                continue
            on_new_rows([row])

# -------------------- Main tracker logic --------------------
def run_tracker(args):
    frame_w = args.frame_w
    frame_h = args.frame_h
    hfov = args.hfov
    vfov = args.vfov
    selection = args.selection

    # Latest observed state per track_id
    latest_tracks = {}  # track_id -> { 'ts': datetime, 'cx':float, 'cy':float, 'w':int, 'h':int, 'conf':float, 'frozen':int, 'frame_idx':int }

    current_target_id = None
    last_yaw_err = 0.0
    last_pitch_err = 0.0
    last_time = time.time()

    # Setup output channel
    ser = None
    sock = None
    udp_addr = None
    if args.mode == "serial":
        if serial is None:
            print("pyserial not available. Install 'pyserial' to use serial mode.")
            return
        ser = serial.Serial(args.serial_port, args.baud, timeout=0.1)
        print(f"[INFO] Serial opened: {args.serial_port}@{args.baud}")
    elif args.mode == "udp":
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_addr = (args.udp_host, args.udp_port)
        print(f"[INFO] UDP target: {udp_addr}")
    else:
        print("[INFO] Running in 'print' mode; commands will be printed to stdout.")

    def handle_rows(rows):
        nonlocal latest_tracks, current_target_id, last_yaw_err, last_pitch_err, last_time
        now = time.time()
        for row in rows:
            # handle header rows or malformed lines
            if len(row) < 13:
                continue
            # parse columns (guard against whitespace)
            try:
                ts_s = row[0]
                frame_idx = int(row[1])
                track_id = int(row[2])
                x1 = int(float(row[3])); y1 = int(float(row[4]))
                x2 = int(float(row[5])); y2 = int(float(row[6]))
                cx = float(row[7]); cy = float(row[8])
                w = int(float(row[9])); h = int(float(row[10]))
                conf = float(row[11])
                frozen_flag = int(row[12])
                # parse timestamp if needed, but will use system time for staleness
                # ts_dt = datetime.fromisoformat(ts_s) if ts_s else datetime.now(timezone.utc)
            except Exception:
                continue
            latest_tracks[track_id] = {
                'ts': now,
                'frame_idx': frame_idx,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'cx': cx, 'cy': cy, 'w': w, 'h': h, 'conf': conf, 'frozen': frozen_flag
            }

        # remove stale tracks older than STALE_TRACK_SECONDS
        stale_cutoff = time.time() - STALE_TRACK_SECONDS
        stale_keys = [tid for tid, v in latest_tracks.items() if v['ts'] < stale_cutoff]
        for tid in stale_keys:
            del latest_tracks[tid]
            if current_target_id == tid:
                current_target_id = None

        # pick target if none, otherwise try to keep current_target_id
        chosen = None
        if args.fixed_track is not None:
            # explicit track id requested by user
            if args.fixed_track in latest_tracks:
                chosen = args.fixed_track
        else:
            # keep current if still valid and recent
            if current_target_id is not None and current_target_id in latest_tracks:
                # keep target
                chosen = current_target_id
            else:
                # select best by selection strategy
                if latest_tracks:
                    if selection == "largest":
                        # largest area
                        chosen = max(latest_tracks.items(), key=lambda kv: (kv[1]['w'] * kv[1]['h']))[0]
                    elif selection == "highest_conf":
                        chosen = max(latest_tracks.items(), key=lambda kv: kv[1]['conf'])[0]
                    else:
                        chosen = max(latest_tracks.items(), key=lambda kv: (kv[1]['w'] * kv[1]['h']))[0]

        current_target_id = chosen

        # compute control if we have a target
        if current_target_id is None:
            # no target, optionally send zeros or hold
            # For now: send zero motion (or nothing)
            return

        target = latest_tracks.get(current_target_id)
        if target is None:
            return

        # pixel offsets from center (positive dx to the right, positive dy down)
        dx = target['cx'] - (frame_w / 2.0)
        dy = target['cy'] - (frame_h / 2.0)

        # convert to angular error (deg). Note: pitch sign convention may need flip depending gimbal
        yaw_err_deg, pitch_err_deg = pixel_offset_to_angle(dx, dy, frame_w, frame_h, hfov, vfov)

        # PD control (discrete)
        tnow = time.time()
        dt = max(1e-3, tnow - last_time)
        last_time = tnow

        # derivative of error
        d_yaw = (yaw_err_deg - last_yaw_err) / dt
        d_pitch = (pitch_err_deg - last_pitch_err) / dt

        yaw_cmd = KP_YAW * yaw_err_deg + KD_YAW * d_yaw
        pitch_cmd = KP_PITCH * pitch_err_deg + KD_PITCH * d_pitch

        # clamp commands to max speeds
        yaw_cmd = clamp(yaw_cmd, -MAX_YAW_SPEED_DEG, MAX_YAW_SPEED_DEG)
        pitch_cmd = clamp(pitch_cmd, -MAX_PITCH_SPEED_DEG, MAX_PITCH_SPEED_DEG)

        last_yaw_err = yaw_err_deg
        last_pitch_err = pitch_err_deg

        # If frozen send zeroes.
        if target.get('frozen', 0) == 1:
            yaw_cmd = 0.0
            pitch_cmd = 0.0

        # dispatch command according to mode
        if args.mode == "serial" and ser is not None:
            send_command_serial(ser, yaw_cmd, pitch_cmd)
        elif args.mode == "udp" and sock is not None:
            send_command_udp(sock, udp_addr, yaw_cmd, pitch_cmd)
        else:
            send_command_print(yaw_cmd, pitch_cmd)

    # Start tailing the CSV
    if not os.path.exists(args.csv):
        print(f"[ERROR] CSV file not found: {args.csv}")
        return

    print(f"[INFO] Tailing csv: {args.csv}, mode={args.mode}, hfov={hfov}, vfov={vfov}, frame={frame_w}x{frame_h}")
    try:
        tail_csv_realtime(args.csv, handle_rows)
    except KeyboardInterrupt:
        print("[INFO] Exiting on user interrupt")
    finally:
        if ser:
            ser.close()
        if sock:
            sock.close()

# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser(description="Real-time gimbal tracker reading tracks_log.csv")
    p.add_argument("--csv", default=DEFAULT_CSV, help="Path to tracks_log.csv")
    p.add_argument("--mode", choices=["print", "serial", "udp"], default="print",
                   help="How to send commands: print (debug), serial, or udp")
    p.add_argument("--serial-port", default="COM3", help="Serial port (for serial mode)")
    p.add_argument("--baud", type=int, default=115200, help="Serial baud")
    p.add_argument("--udp-host", default="127.0.0.1", help="UDP host")
    p.add_argument("--udp-port", type=int, default=9000, help="UDP port")
    p.add_argument("--hfov", type=float, default=DEFAULT_HFOV_DEG, help="Camera horizontal FOV (deg)")
    p.add_argument("--vfov", type=float, default=DEFAULT_VFOV_DEG, help="Camera vertical FOV (deg)")
    p.add_argument("--frame-w", type=int, default=DEFAULT_FRAME_W, help="Camera frame width in pixels")
    p.add_argument("--frame-h", type=int, default=DEFAULT_FRAME_H, help="Camera frame height in pixels")
    p.add_argument("--selection", choices=["largest", "highest_conf"], default=DEFAULT_SELECTION,
                   help="Choose which face to track when multiple are present")
    p.add_argument("--fixed-track", type=int, default=None,
                   help="If provided, will follow this track_id specifically (if present)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_tracker(args)

