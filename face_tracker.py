import cv2
import mediapipe as mp
import numpy as np

class FaceTracker:
    def __init__(self, static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    def get_landmarks(self, frame):
        # Convert the BGR image to RGB before processing.
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if results.multi_face_landmarks:
            # We only care about the first face for this project
            face_landmarks = results.multi_face_landmarks[0]
            h, w, c = frame.shape
            landmarks = []
            for lm in face_landmarks.landmark:
                landmarks.append((int(lm.x * w), int(lm.y * h)))
            return landmarks
        return None

    def draw_landmarks(self, frame, landmarks):
        if landmarks:
            for pt in landmarks:
                cv2.circle(frame, pt, 1, (0, 255, 0), -1)
        return frame
