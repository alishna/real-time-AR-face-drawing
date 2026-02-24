import cv2
import mediapipe as mp
import numpy as np
import time
import json

# Controls:
# d - toggle draw mode
# c - cycle color
# + / - - brush size
# x - clear strokes
# s - save snapshot
# v - toggle video record
# q / ESC - quit

COLORS = [(0,255,255),(0,128,255),(0,0,255),(0,255,0),(255,0,0),(255,255,255),(0,0,0)]

# Global state used by mouse callback
current_face_center = None  # (x_norm, y_norm)
current_face_width = None   # width in normalized coords
drawing_enabled = False
brush_size = 3
color_index = 0
strokes = []  # list of strokes; each stroke is dict: {'color':(b,g,r),'size':int,'points':[(nx,ny),...]}
_current_stroke = None
frame_w = 640
frame_h = 480

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=False,
                             min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)

recording = False
video_writer = None
fourcc = cv2.VideoWriter_fourcc(*'XVID')

last_time = 0


def face_metrics_from_landmarks(landmarks):
    # landmarks: list of landmark with .x and .y in normalized coords
    xs = np.array([lm.x for lm in landmarks])
    ys = np.array([lm.y for lm in landmarks])
    cx = float(xs.mean())
    cy = float(ys.mean())
    width = float(xs.max() - xs.min())
    width = max(width, 0.001)
    return (cx, cy), width


def transform_norm_to_pixel(nx, ny, face_center, face_width, fw, fh):
    # nx,ny are normalized relative coords stored as (x_rel, y_rel) where x_rel = (x_norm - center_x)/face_width
    x_norm = face_center[0] + nx * face_width
    y_norm = face_center[1] + ny * face_width
    px = int(np.clip(x_norm * fw, 0, fw-1))
    py = int(np.clip(y_norm * fh, 0, fh-1))
    return px, py


def mouse_callback(event, x, y, flags, param):
    global _current_stroke, strokes
    global current_face_center, current_face_width
    global drawing_enabled, brush_size, color_index

    if not drawing_enabled:
        return
    if current_face_center is None or current_face_width is None:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        _current_stroke = {'color': COLORS[color_index], 'size': brush_size, 'points': []}
        # fallthrough to add first point
    if event == cv2.EVENT_MOUSEMOVE and (_current_stroke is not None) and (flags & cv2.EVENT_FLAG_LBUTTON):
        # convert pixel (x,y) to normalized relative coords
        x_norm = x / frame_w
        y_norm = y / frame_h
        nx = (x_norm - current_face_center[0]) / current_face_width
        ny = (y_norm - current_face_center[1]) / current_face_width
        _current_stroke['points'].append((nx, ny))
    if event == cv2.EVENT_LBUTTONUP and _current_stroke is not None:
        # finalize stroke
        if len(_current_stroke['points']) > 0:
            strokes.append(_current_stroke)
        _current_stroke = None


cv2.namedWindow('AR Draw')
cv2.setMouseCallback('AR Draw', mouse_callback)

print("Starting AR drawing. Press 'd' to toggle drawing mode. 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    fh, fw = frame.shape[:2]

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    face_present = False
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        (cx, cy), fwidth = face_metrics_from_landmarks(landmarks)
        current_face_center = (cx, cy)
        current_face_width = fwidth
        face_present = True
        # optional: draw mesh points
        # for lm in landmarks:
        #     px = int(lm.x * fw)
        #     py = int(lm.y * fh)
        #     cv2.circle(frame, (px,py), 1, (0,255,0), -1)
    else:
        current_face_center = None
        current_face_width = None

    overlay = frame.copy()

    # render strokes anchored to face
    if face_present:
        for stroke in strokes:
            pts = stroke['points']
            if not pts:
                continue
            prev = None
            for p in pts:
                px, py = transform_norm_to_pixel(p[0], p[1], current_face_center, current_face_width, fw, fh)
                if prev is not None:
                    cv2.line(overlay, prev, (px,py), stroke['color'], stroke['size'], cv2.LINE_AA)
                else:
                    cv2.circle(overlay, (px,py), stroke['size']//2 + 1, stroke['color'], -1)
                prev = (px,py)
        # current stroke in progress
        if _current_stroke is not None and _current_stroke['points']:
            prev = None
            for p in _current_stroke['points']:
                px, py = transform_norm_to_pixel(p[0], p[1], current_face_center, current_face_width, fw, fh)
                if prev is not None:
                    cv2.line(overlay, prev, (px,py), _current_stroke['color'], _current_stroke['size'], cv2.LINE_AA)
                else:
                    cv2.circle(overlay, (px,py), _current_stroke['size']//2 + 1, _current_stroke['color'], -1)
                prev = (px,py)
    else:
        # when face not present, optionally show strokes at last-known position faded
        pass

    # combine overlay
    alpha = 0.9
    frame_out = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)

    # HUD
    now = time.time()
    fps = 1/(now-last_time) if last_time else 0
    last_time = now
    status = f"Draw:{'ON' if drawing_enabled else 'OFF'}  Color:{color_index+1}/{len(COLORS)}  Brush:{brush_size}  FPS:{int(fps)}"
    cv2.putText(frame_out, status, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

    cv2.imshow('AR Draw', frame_out)

    if recording and video_writer is None:
        video_writer = cv2.VideoWriter(f'record_{int(time.time())}.avi', fourcc, 20.0, (fw, fh))
    if recording and video_writer is not None:
        video_writer.write(frame_out)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('d'):
        drawing_enabled = not drawing_enabled
        print('Drawing:', drawing_enabled)
    elif key == ord('c'):
        color_index = (color_index + 1) % len(COLORS)
        print('Color index:', color_index)
    elif key == ord('+') or key == ord('='):
        brush_size += 1
    elif key == ord('-'):
        brush_size = max(1, brush_size-1)
    elif key == ord('x'):
        strokes = []
        print('Cleared strokes')
    elif key == ord('s'):
        fname = f'snapshot_{int(time.time())}.png'
        cv2.imwrite(fname, frame_out)
        print('Saved', fname)
    elif key == ord('v'):
        recording = not recording
        if not recording and video_writer is not None:
            video_writer.release()
            video_writer = None
            print('Saved recording')
        else:
            print('Recording started')
    elif key == 27 or key == ord('q'):
        break

# cleanup
cap.release()
if video_writer is not None:
    video_writer.release()
cv2.destroyAllWindows()
face_mesh.close()

# optionally persist strokes
with open('strokes.json', 'w') as f:
    json.dump(strokes, f)

print('Exiting')
