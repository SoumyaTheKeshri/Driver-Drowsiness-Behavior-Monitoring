"""
VigiDrive - AI Driver Monitoring System
Main entry point
"""

import cv2
import time
import argparse
from core.detector import DriverMonitor
from core.alert import AlertSystem
from utils.display import Dashboard

def parse_args():
    parser = argparse.ArgumentParser(description="VigiDrive - AI Driver Monitoring System")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--no-display", action="store_true", help="Run without display (headless)")
    parser.add_argument("--save-log", action="store_true", help="Save event log to file")
    parser.add_argument("--sensitivity", type=str, default="medium",
                        choices=["low", "medium", "high"],
                        help="Detection sensitivity (default: medium)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 50)
    print("   VigiDrive - AI Driver Monitoring System")
    print("=" * 50)
    print(f"[INFO] Camera Index : {args.camera}")
    print(f"[INFO] Sensitivity  : {args.sensitivity}")
    print(f"[INFO] Save Log     : {args.save_log}")
    print("[INFO] Initializing system...")

    # Initialize components
    monitor   = DriverMonitor(sensitivity=args.sensitivity)
    alert_sys = AlertSystem(save_log=args.save_log)
    dashboard = Dashboard()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Check camera index.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("[INFO] System ready. Press 'q' to quit.\n")

    fps_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        frame_count += 1

        # Core analysis
        result = monitor.analyze(frame)

        # Trigger alerts based on result
        alert_sys.process(result)

        if not args.no_display:
            # Draw overlay on frame
            display_frame = dashboard.render(frame, result, alert_sys.get_active_alerts())

            # FPS counter
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_time)
                fps_time = time.time()
                dashboard.fps = fps

            cv2.imshow("VigiDrive Monitor", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[INFO] Shutting down VigiDrive...")
            break

    cap.release()
    cv2.destroyAllWindows()
    alert_sys.close()
    print("[INFO] Session ended.")


if __name__ == "__main__":
    main()
