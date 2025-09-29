"""
Simple UDP helper: listens for face_detector.py UDP telemetry and converts pixel -> yaw/pitch angles.
Usage example:
  python udp_gimbal_helper.py --listen-port 22345 --use-fov --hfov 62 --vfov 44
"""
import socket, json, argparse, time, math

# convert pixel offset to angle (using HFOV/VFOV approx)
def pixel_offset_to_angle(dx_pixels, dy_pixels, frame_w, frame_h, hfov_deg, vfov_deg):
    nx = dx_pixels / (frame_w / 2.0)
    ny = dy_pixels / (frame_h / 2.0)
    yaw_deg = nx * (hfov_deg / 2.0)
    pitch_deg = ny * (vfov_deg / 2.0)
    return yaw_deg, pitch_deg

# using intrinsics: approximate angle = arctan((x - cx)/fx)
def pixel_to_angle_intrinsics(cx_pix, cy_pix, frame_w, frame_h, fx, fy, cx, cy):
    dx = (cx_pix - cx) / fx
    dy = (cy_pix - cy) / fy
    yaw = math.degrees(math.atan(dx))
    pitch = math.degrees(math.atan(dy))
    return yaw, pitch

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--listen-host", default="0.0.0.0")
    p.add_argument("--listen-port", type=int, default=22345)
    p.add_argument("--frame-w", type=int, default=640)
    p.add_argument("--frame-h", type=int, default=480)
    p.add_argument("--use-fov", action="store_true", help="Use HFOV/VFOV approximation")
    p.add_argument("--hfov", type=float, default=62.0)
    p.add_argument("--vfov", type=float, default=44.0)
    p.add_argument("--use-intrinsics", action="store_true")
    p.add_argument("--fx", type=float, default=640.0)
    p.add_argument("--fy", type=float, default=640.0)
    p.add_argument("--cx", type=float, default=320.0)
    p.add_argument("--cy", type=float, default=240.0)
    p.add_argument("--mode", choices=["print","udp"], default="print", help="Output mode")
    p.add_argument("--out-udp-host", default="127.0.0.1")
    p.add_argument("--out-udp-port", type=int, default=9000)
    args = p.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.listen_host, args.listen_port))
    print(f"Listening {args.listen_host}:{args.listen_port} ...")

    out_sock = None
    out_addr = None
    if args.mode == "udp":
        out_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        out_addr = (args.out_udp_host, args.out_udp_port)
        print("Will send commands via UDP to", out_addr)

    while True:
        try:
            data, addr = sock.recvfrom(65536)
            txt = data.decode('utf-8', errors='ignore')
            payload = json.loads(txt)
            tracks = payload.get("tracks", [])
            # simple selection: choose largest area or highest conf
            chosen = None
            best_score = -1.0
            for t in tracks:
                w = int(t.get("w",0)); h = int(t.get("h",0))
                conf = float(t.get("conf",0.0))
                score = w*h  # prefer largest
                if score > best_score:
                    best_score = score
                    chosen = t
            if chosen is None:
                continue
            cx = int(chosen.get("cx")); cy = int(chosen.get("cy"))
            if args.use_introninsics:
                yaw_deg, pitch_deg = pixel_to_angle_intrinsics(cx, cy, args.frame_w, args.frame_h, args.fx, args.fy, args.cx, args.cy)
            elif args.use_fov:
                yaw_deg, pitch_deg = pixel_offset_to_angle(cx - (args.frame_w/2.0), cy - (args.frame_h/2.0), args.frame_w, args.frame_h, args.hfov, args.vfov)
            else:
                yaw_deg, pitch_deg = pixel_offset_to_angle(cx - (args.frame_w/2.0), cy - (args.frame_h/2.0), args.frame_w, args.frame_h, args.hfov, args.vfov)
            out_msg = f"{yaw_deg:.3f},{pitch_deg:.3f}"
            if args.mode == "print":
                print(f"[{time.time():.3f}] -> yaw:{yaw_deg:.3f} deg  pitch:{pitch_deg:.3f} deg  (track id={chosen.get('id')})")
            elif args.mode == "udp" and out_sock is not None:
                out_sock.sendto(out_msg.encode('ascii'), out_addr)
        except KeyboardInterrupt:
            break
        except Exception as e:
            # ignore malformed packets
            continue

if __name__ == "__main__":
    main()
