import cv2
import numpy as np

def resize_keep_aspect(image, target_w, target_h):
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((target_h, target_w, 3), dtype=resized.dtype)
    x0 = (target_w - nw) // 2
    y0 = (target_h - nh) // 2
    out[y0:y0+nh, x0:x0+nw] = resized
    return out
