# Draw in Real-Time AR

Create interactive augmented reality accessories directly on your face using a live webcam feed. This application uses MediaPipe Face Mesh for high-precision tracking and OpenCV for rendering.

## Key Features
- **Real-Time Landmark Anchoring**: Drawings scale and move with your face.
- **Custom Drawing Interface**: Use your mouse to design jewelry, hats, and more.
- **Floating Control Panel**: Adjust brush size, color, or save screenshots via a dedicated GUI.

## Installation

1. Clone or download this project.
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main application:
```powershell
python main.py
```

### Controls
- **Left Mouse Click & Drag**: Draw on the video window.
- **Control Window**: Use buttons to change color, brush size, or clear the canvas.
- **'q' key**: Quit the application (press while the video window is focused).
- **'c' key**: Quick-clear all drawings.

## Technical Stack
- Python 3.10+
- MediaPipe (Face Mesh)
- OpenCV
- NumPy
- Tkinter (GUI)

