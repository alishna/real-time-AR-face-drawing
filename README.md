# Draw Your Own Hats & Jewelry in Real-Time AR

This project is a lightweight Python app that lets users draw custom accessories (hats, glasses, jewelry) onto their face in real-time using a webcam and MediaPipe Face Mesh. Drawings are anchored to facial landmarks and follow head movement.

## Requirements

Install packages (preferably in a virtualenv):

```bash
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

## Controls

- `d` - Toggle drawing mode (use mouse to draw while enabled)
- `c` - Cycle color
- `+` / `-` - Increase / decrease brush size
- `x` - Clear strokes
- `s` - Save snapshot
- `v` - Toggle video recording
- `q` or `Esc` - Quit

Notes:
- Draw while a face is detected so strokes anchor correctly.
- Strokes are saved to `strokes.json` on exit.

## Extensions

- Add color picker UI with Tkinter or PyQt
- Add animated effects or expression-triggered actions
- Save/share short recorded clips

