import cv2
import numpy as np
import tkinter as tk
from threading import Thread
from face_tracker import FaceTracker
from drawing_tool import DrawingTool

# Global state for mouse handling
drawing = False
current_landmarks = None
latest_frame = None
drawer = DrawingTool()

def mouse_callback(event, x, y, flags, param):
    global drawing, current_landmarks, drawer
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        drawer.start_stroke()
        if current_landmarks:
            drawer.add_point(x, y)
            
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing and current_landmarks:
            drawer.add_point(x, y)
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if current_landmarks:
            drawer.end_stroke(current_landmarks)

def create_gui():
    global drawer
    root = tk.Tk()
    root.title("AR Accessories Controls")
    root.geometry("300x400")
    root.configure(bg='#2c3e50')

    label = tk.Label(root, text="Brush Controls", font=("Helvetica", 16, "bold"), bg='#2c3e50', fg='white')
    label.pack(pady=10)

    # Color Selection
    colors = [
        ("Red", (0, 0, 255)),
        ("Green", (0, 255, 0)),
        ("Blue", (255, 0, 0)),
        ("Yellow", (0, 255, 255)),
        ("Magenta", (255, 0, 255)),
        ("Cyan", (255, 255, 0)),
        ("White", (255, 255, 255)),
        ("Black", (0, 0, 0))
    ]

    color_frame = tk.Frame(root, bg='#2c3e50')
    color_frame.pack(pady=10)

    def set_color(c):
        drawer.color = c

    for name, bgr in colors:
        btn = tk.Button(color_frame, text=name, bg='#34495e', fg='white', width=10, 
                       command=lambda c=bgr: set_color(c))
        btn.pack(side=tk.TOP, pady=2)

    # Thickness Slider
    thick_label = tk.Label(root, text="Brush Size", bg='#2c3e50', fg='white')
    thick_label.pack(pady=(10, 0))
    
    def set_thick(val):
        drawer.thickness = int(val)

    slider = tk.Scale(root, from_=1, to=20, orient=tk.HORIZONTAL, bg='#2c3e50', fg='white', 
                     highlightthickness=0, command=set_thick)
    slider.set(drawer.thickness)
    slider.pack(pady=10, fill=tk.X, padx=20)

    # Screenshot Button
    def save_screenshot():
        global latest_frame
        if latest_frame is not None:
            import time
            filename = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(filename, latest_frame)
            print(f"Saved screenshot: {filename}")
    
    shot_btn = tk.Button(root, text="Save Screenshot", bg='#2980b9', fg='white', font=("Helvetica", 12, "bold"),
                        command=save_screenshot)
    shot_btn.pack(pady=10, fill=tk.X, padx=20)

    # Clear Button
    def clear_canvas():
        drawer.strokes = []
    
    clear_btn = tk.Button(root, text="Clear All", bg='#e74c3c', fg='white', font=("Helvetica", 12, "bold"),
                         command=clear_canvas)
    clear_btn.pack(pady=10, fill=tk.X, padx=20)

    root.mainloop()

def main():
    global current_landmarks, latest_frame
    latest_frame = None
    
    # Start GUI in a separate thread
    gui_thread = Thread(target=create_gui, daemon=True)
    gui_thread.start()

    cap = cv2.VideoCapture(0)
    tracker = FaceTracker()
    
    cv2.namedWindow("AR Face Drawing")
    cv2.setMouseCallback("AR Face Drawing", mouse_callback)
    
    print("Application Started.")
    print("Draw on the video feed window. Use the control panel to change settings.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        current_landmarks = tracker.get_landmarks(frame)
        
        if current_landmarks:
            strokes = drawer.get_projected_strokes(current_landmarks)
            for stroke in strokes:
                pts = np.array(stroke['points'], np.int32)
                if len(pts) > 1:
                    cv2.polylines(frame, [pts], False, stroke['color'], stroke['thickness'])
        
        # Draw current stroke (real-time feedback)
        if drawing and len(drawer.current_stroke) > 1:
            pts = np.array(drawer.current_stroke, np.int32)
            cv2.polylines(frame, [pts], False, drawer.color, drawer.thickness)

        # UI Overlay on OpenCV frame
        cv2.putText(frame, "AR Drawing Mode", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        latest_frame = frame.copy()
        cv2.imshow("AR Face Drawing", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            drawer.strokes = []

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
