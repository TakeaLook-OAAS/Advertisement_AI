import cv2
import numpy as np

def draw_overlay(frame_bgr, result, *, fps: float, cfg):
    disp = cfg.get("display", {})
    font_scale = float(disp.get("font_scale", 0.7))
    thickness = int(disp.get("thickness", 2))

    vis = frame_bgr.copy()

    y = 30
    if disp.get("draw_fps", True):
        cv2.putText(vis, f"FPS: {fps:.2f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
        y += 30

    if disp.get("draw_headpose", True) and result.headpose:
        hp = result.headpose
        cv2.putText(vis, f"Yaw: {hp['yaw']:.1f} Pitch: {hp['pitch']:.1f} Roll: {hp['roll']:.1f}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        y += 30

    if disp.get("draw_gaze", True) and result.gaze is not None:
        gv = result.gaze
        cv2.putText(vis, f"GazeVec: [{gv[0]:.2f}, {gv[1]:.2f}, {gv[2]:.2f}]",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        y += 30

    if result.attending is not None:
        status = "ATTENDING" if result.attending else "NOT"
        cv2.putText(vis, status, (10, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)

    # ROI polygon if provided
    poly = (result.meta or {}).get("roi_polygon")
    if poly:
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

    return vis
