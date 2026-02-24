import numpy as np
import cv2

class DrawingTool:
    def __init__(self):
        self.strokes = []  # List of { 'points': [ (x, y), ... ], 'color': (b, g, r), 'thickness': int, 'ref_indices': (i, j, k), 'rel_coords': [ (u, v), ... ] }
        self.current_stroke = []
        self.color = (0, 0, 255)  # Default red
        self.thickness = 2
        
        # Standard landmark indices for anchoring (MediaPipe Face Mesh)
        # 1: Nose tip
        # 33: Left eye outer
        # 263: Right eye outer
        # 152: Chin
        # 10: Forehead top
        self.anchor_indices = (33, 263, 152) # Triangle across face

    def start_stroke(self, color=None, thickness=None):
        if color: self.color = color
        if thickness: self.thickness = thickness
        self.current_stroke = []

    def add_point(self, x, y):
        self.current_stroke.append((x, y))

    def end_stroke(self, landmarks):
        if not self.current_stroke or landmarks is None:
            self.current_stroke = []
            return

        # Calculate relative coordinates for the stroke points based on anchor landmarks
        rel_coords = []
        p1 = np.array(landmarks[self.anchor_indices[0]])
        p2 = np.array(landmarks[self.anchor_indices[1]])
        p3 = np.array(landmarks[self.anchor_indices[2]])

        # Create a basis for the affine transform
        # We'll use a simple approach: find the transformation matrix from the triangle to a unit triangle
        # or just store barycentric coordinates.
        # Let's use an affine transform matrix approach for simplicity in reconstruction.
        
        src_tri = np.array([p1, p2, p3], dtype=np.float32)
        # Reference triangle in normalized space (arbitrary, say 0,0 1,0 0,1)
        dst_tri = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
        
        matrix = cv2.getAffineTransform(src_tri, dst_tri)
        
        for pt in self.current_stroke:
            # Transform point to reference space
            p = np.array([pt[0], pt[1], 1], dtype=np.float32)
            rel_pt = matrix @ p
            rel_coords.append((rel_pt[0], rel_pt[1]))

        self.strokes.append({
            'rel_coords': rel_coords,
            'color': self.color,
            'thickness': self.thickness,
            'anchor_indices': self.anchor_indices
        })
        self.current_stroke = []

    def get_projected_strokes(self, landmarks):
        if landmarks is None:
            return []

        projected_strokes = []
        p1 = np.array(landmarks[self.anchor_indices[0]], dtype=np.float32)
        p2 = np.array(landmarks[self.anchor_indices[1]], dtype=np.float32)
        p3 = np.array(landmarks[self.anchor_indices[2]], dtype=np.float32)

        # Map from reference space back to current face space
        src_tri = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
        dst_tri = np.array([p1, p2, p3], dtype=np.float32)
        
        matrix = cv2.getAffineTransform(src_tri, dst_tri)

        for stroke in self.strokes:
            pts = []
            for rel_pt in stroke['rel_coords']:
                p = np.array([rel_pt[0], rel_pt[1], 1], dtype=np.float32)
                proj_pt = matrix @ p
                pts.append((int(proj_pt[0]), int(proj_pt[1])))
            
            projected_strokes.append({
                'points': pts,
                'color': stroke['color'],
                'thickness': stroke['thickness']
            })
        
        return projected_strokes
