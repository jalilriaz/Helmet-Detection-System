"""Automated Motorcycle Helmet Detection System.

High-performance real-time detection optimized for CPU and edge devices.
Features:
- Dual model support: YOLOv11 Nano (ultra-fast, ~30 FPS on CPU) and YOLOv8 Medium (high precision).
- Threaded asynchronous webcam capture to eliminate video lag and buffer delay.
- Intelligent Pedestrian Filtering: Ignores pedestrians and only alerts on motorcycle/bicycle riders.
- Real-time performance HUD: Live FPS and latency overlay.
- Automatic violation snapshot recording.
- Full Gradio UI supporting Photo check, Live browser camera, and Recorded Video Analysis (.mp4).

Usage:
    python app.py                       # Launch Gradio Web UI
    python app.py --webcam              # Launch fast OpenCV desktop webcam
    python app.py --webcam --model nano # Force YOLOv11 Nano (fastest)
    python app.py --webcam --imgsz 320  # Use 320px inference for maximum speed
    python app.py --webcam --camera 1   # Use external USB camera
"""

import os
import sys
import time
import datetime
import inspect
import threading
import argparse

import cv2
import numpy as np
from PIL import Image, ImageOps
import gradio as gr
from ultralytics import YOLO

# Cache directory configuration
os.environ['TORCH_HOME'] = '/tmp/torch_cache'
os.environ['HF_HOME'] = '/tmp/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'
os.environ['TMPDIR'] = '/tmp'

# Available trained helmet models
AVAILABLE_MODELS = {
    "medium": ("best.pt", "YOLOv8 Medium (High Precision - Best for Photos)"),
    "nano": ("yolov11nbest.pt", "YOLOv11 Nano (Fastest - Best for Real-Time Webcam)"),
}
# Filter to existing files
AVAILABLE_MODELS = {k: v for k, v in AVAILABLE_MODELS.items() if os.path.exists(v[0])}

if not AVAILABLE_MODELS:
    raise FileNotFoundError("No model weights found. Ensure 'best.pt' or 'yolov11nbest.pt' is in the project folder.")

# Model instances cache
LOADED_MODELS = {}
MODEL_LOCK = threading.Lock()
ACTIVE_MODEL_KEY = "medium" if "medium" in AVAILABLE_MODELS else list(AVAILABLE_MODELS.keys())[0]


def get_model(key="medium"):
    """Load and cache the requested model, resolving keys, descriptions, or filenames."""
    global ACTIVE_MODEL_KEY
    resolved_key = None
    key_str = str(key).strip().lower()

    if key in AVAILABLE_MODELS:
        resolved_key = key
    else:
        for k, (path, desc) in AVAILABLE_MODELS.items():
            if key == desc or k in key_str or path in key_str or ("medium" in key_str and k == "medium") or ("nano" in key_str and k == "nano") or ("v8" in key_str and k == "medium") or ("v11" in key_str and k == "nano"):
                resolved_key = k
                break

    if resolved_key is None:
        resolved_key = "medium" if "medium" in AVAILABLE_MODELS else list(AVAILABLE_MODELS.keys())[0]

    with MODEL_LOCK:
        if resolved_key not in LOADED_MODELS:
            path, desc = AVAILABLE_MODELS[resolved_key]
            print(f"✓ Loading model: {desc} from {path}...")
            LOADED_MODELS[resolved_key] = YOLO(path)
        ACTIVE_MODEL_KEY = resolved_key
        return LOADED_MODELS[resolved_key], AVAILABLE_MODELS[resolved_key][1]


# Pre-load default model
CURRENT_MODEL, CURRENT_MODEL_NAME = get_model(ACTIVE_MODEL_KEY)

# Class name sets
HELMET_NAMES = {
    'accepthelmet', 'helmet', 'cyclinghelmet', 'halfface', 'hardhat',
    'modularhelmet', 'nutshell', 'quarterfacehelmet', 'sportshelmet',
}
NO_HELMET_NAMES = {'nonhelmet', 'nohelmet', 'withouthelmet', 'nothelmet'}
INFERENCE_LOCK = threading.Lock()

# Directory for violation evidence
VIOLATIONS_DIR = "violations"
LAST_VIOLATION_TIME = 0
VIOLATION_LOCK = threading.Lock()

# Companion detector for pedestrian suppression
COMPANION_DETECTOR = None
COMPANION_LOCK = threading.Lock()


def get_companion_detector():
    """Lazily load standard COCO YOLO model to detect motorcycles/bicycles for pedestrian filtering."""
    global COMPANION_DETECTOR
    if COMPANION_DETECTOR is None:
        with COMPANION_LOCK:
            if COMPANION_DETECTOR is None:
                for base in ["yolo11n.pt", "yolov8n.pt"]:
                    try:
                        print(f"Loading vehicle detector ({base}) for pedestrian filtering...")
                        COMPANION_DETECTOR = YOLO(base)
                        break
                    except Exception:
                        continue
    return COMPANION_DETECTOR


def save_violation_evidence(frame_bgr, count):
    """Save an evidence snapshot of a confirmed no-helmet violation with rate limiting."""
    global LAST_VIOLATION_TIME
    now = time.time()
    with VIOLATION_LOCK:
        # At most 1 snapshot every 2.0 seconds to prevent disk flooding
        if now - LAST_VIOLATION_TIME < 2.0:
            return None
        LAST_VIOLATION_TIME = now
        os.makedirs(VIOLATIONS_DIR, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(VIOLATIONS_DIR, f"violation_{timestamp}_{count}nohelmet.jpg")
        cv2.imwrite(filename, frame_bgr)
        print(f"📸 [VIOLATION SAVED] {filename}")
        return filename


def display_class(raw_name):
    key = ''.join(c for c in str(raw_name).lower() if c.isalnum())
    if key in NO_HELMET_NAMES:
        return 'No Helmet', (220, 38, 38)     # Red
    if key in HELMET_NAMES:
        return 'Helmet', (22, 163, 74)        # Green
    return str(raw_name), (37, 99, 235)       # Blue


def is_head_on_bike(head_box, bike_boxes):
    """Determine if a head detection is associated with a motorcycle/bicycle (not a pedestrian)."""
    if not bike_boxes:
        return False
    hx1, hy1, hx2, hy2 = head_box
    head_cx = (hx1 + hx2) / 2.0
    for bx1, by1, bx2, by2 in bike_boxes:
        # Check horizontal overlap and that bike is below or overlapping head
        bike_w = bx2 - bx1
        expanded_bx1 = bx1 - bike_w * 0.25
        expanded_bx2 = bx2 + bike_w * 0.25
        if expanded_bx1 <= head_cx <= expanded_bx2:
            # Head should be near the top of the bike or slightly above it
            if by1 <= hy2 + (by2 - by1) * 0.6 and by2 >= hy1:
                return True
    return False


def draw_detection(canvas, xyxy, text, colour):
    """Draw bounding box and label directly on an RGB or BGR numpy array."""
    height, width = canvas.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy]
    x1, x2 = sorted((max(0, min(x1, width - 1)), max(0, min(x2, width - 1))))
    y1, y2 = sorted((max(0, min(y1, height - 1)), max(0, min(y2, height - 1))))
    thickness = max(1, min(3, round(min(width, height) / 350)))
    scale = max(0.4, min(0.8, min(width, height) / 900))
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, 1)
    if tw + 10 > width:
        scale *= max(1, width - 10) / (tw + 1)
        (tw, th), baseline = cv2.getTextSize(text, font, scale, 1)
    left = max(0, min(x1, width - tw - 12))
    label_height = th + baseline + 8
    top = max(0, min(y1 - label_height if y1 >= label_height else y1, height - label_height))
    cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, thickness)
    cv2.rectangle(canvas, (left, top), (min(width - 1, left + tw + 10), min(height - 1, top + label_height)), colour, -1)
    cv2.putText(canvas, text, (left + 5, top + th + 3), font, scale, (255, 255, 255), 1, cv2.LINE_AA)


def detect(image, confidence=0.25, model_key="nano", imgsz=320,
           filter_pedestrians=False, save_violations=False):
    """Optimized inference pipeline for photos, live webcam, and video frames.

    Args:
        image: PIL.Image or numpy.ndarray (RGB or BGR).
        confidence: Detection threshold (0.05 - 0.95).
        model_key: 'nano' for YOLOv11 (fast) or 'medium' for YOLOv8 (precise).
        imgsz: Inference resolution (320 for real-time speed, 640 for static photos).
        filter_pedestrians: When True, ignores pedestrians and only flags motorcycle/bike riders.
        save_violations: When True, saves an evidence photo to violations/ on infractions.
    """
    if image is None:
        return None, 'Add a photograph or start the camera.', [], '', 0.0

    start = time.perf_counter()
    yolo_model, _ = get_model(model_key)

    is_pil = isinstance(image, Image.Image)
    if is_pil:
        pil_image = ImageOps.exif_transpose(image).convert('RGB')
        canvas = np.asarray(pil_image).copy()
        # Pass PIL image directly to YOLO so it recognizes standard RGB without channel inversion
        yolo_input = pil_image
    else:
        canvas = np.asarray(image).copy()
        yolo_input = canvas  # numpy array from OpenCV (BGR)

    h, w = canvas.shape[:2]

    # Optional: Detect motorcycles/bicycles to associate heads with riders
    bike_boxes = []
    if filter_pedestrians:
        companion = get_companion_detector()
        if companion is not None:
            try:
                # COCO classes: 1 = bicycle, 3 = motorcycle
                c_res = companion(yolo_input, classes=[1, 3], conf=0.25, imgsz=imgsz, verbose=False)[0]
                if c_res.boxes is not None:
                    bike_boxes = [box.tolist() for box in c_res.boxes.xyxy.cpu()]
            except Exception as e:
                print(f"Vehicle detection error: {e}")

    # Run helmet inference
    with INFERENCE_LOCK:
        result = yolo_model(yolo_input, conf=float(confidence), imgsz=int(imgsz), iou=0.6, verbose=False)[0]
        entries = [] if result.boxes is None else result.boxes.data.cpu().tolist()
        names = result.names

    counts = {'Helmet': 0, 'No Helmet': 0, 'Pedestrians': 0}
    rows = []

    # Sort so high-confidence detections draw on top
    for entry in sorted(entries, key=lambda row: row[-2]):
        xyxy, score, class_id = entry[:4], float(entry[-2]), int(entry[-1])
        raw_label = names[class_id]
        label, colour = display_class(raw_label)

        # Apply pedestrian filtering if enabled
        if filter_pedestrians and label == 'No Helmet':
            on_bike = is_head_on_bike(xyxy, bike_boxes)
            if not on_bike:
                # This is a pedestrian on the street, not a motorcycle rider!
                counts['Pedestrians'] += 1
                rows.append(['Pedestrian (Ignored)', round(score * 100, 1)])
                # Draw in neutral gray
                draw_detection(canvas, xyxy, f'Pedestrian {score:.0%}', (120, 144, 156))
                continue

        if label in counts:
            counts[label] += 1
        rows.append([label, round(score * 100, 1)])
        draw_detection(canvas, xyxy, f'{label} {score:.0%}', colour)

    rows.sort(key=lambda row: row[1], reverse=True)

    # Save violation evidence if enabled
    if save_violations and counts['No Helmet'] > 0:
        bgr_frame = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR) if len(canvas.shape) == 3 else canvas
        save_violation_evidence(bgr_frame, counts['No Helmet'])

    elapsed = time.perf_counter() - start
    fps = 1.0 / elapsed if elapsed > 0 else 0.0
    latency_ms = elapsed * 1000

    if counts['No Helmet']:
        status = (f"🚨 WARNING: No Helmet detected: {counts['No Helmet']} | "
                  f"Helmet: {counts['Helmet']}")
    elif counts['Helmet']:
        status = f"✅ Compliant: {counts['Helmet']} Helmet(s) detected | No Helmet: 0"
    else:
        status = 'No helmet detections in current view.'

    if filter_pedestrians and counts['Pedestrians']:
        status += f" (Filtered {counts['Pedestrians']} pedestrian(s))"

    timing = f'Speed: {fps:.1f} FPS | Latency: {latency_ms:.1f} ms | Resolution: {imgsz}px'

    out_image = Image.fromarray(canvas) if isinstance(image, Image.Image) else canvas
    return out_image, status, rows, timing, fps


# ---------------------------------------------------------------------------
# Threaded Camera for Zero-Lag OpenCV Desktop Capture
# ---------------------------------------------------------------------------
class ThreadedCamera:
    """Asynchronous webcam reader running in a daemon thread.
    Continuously discards old frames so capture.read() always returns
    the absolute newest frame, eliminating buffer lag and freeze completely.
    """
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.status, self.frame = self.capture.read()
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            if self.capture.isOpened():
                status, frame = self.capture.read()
                if status:
                    with self.lock:
                        self.status, self.frame = status, frame
            time.sleep(0.005)

    def read(self):
        with self.lock:
            return self.status, (self.frame.copy() if self.frame is not None else None)

    def isOpened(self):
        return self.capture.isOpened()

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.capture.release()


def run_opencv_camera(camera_index=0, model_key="nano", imgsz=320, filter_pedestrians=False):
    """Run real-time detection on local desktop webcam with interactive HUD."""
    cam = ThreadedCamera(camera_index)
    window = 'Automated Helmet Detection (Press Q to quit | P: Pedestrian Filter | M: Switch Model)'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    save_violations = True
    print(f"\n🚀 Starting Live Webcam on camera index {camera_index}...")
    print("Controls:")
    print("  [Q] or [Esc] : Quit")
    print("  [P]          : Toggle Pedestrian Filtering (Focus on Bike Riders)")
    print("  [M]          : Toggle Model (YOLOv11 Nano <-> YOLOv8 Medium)")
    print("  [S]          : Manually capture snapshot\n")

    current_key = model_key
    try:
        if not cam.isOpened():
            raise RuntimeError(f"Cannot open webcam {camera_index}. Check permissions and index.")

        while True:
            ok, bgr = cam.read()
            if not ok or bgr is None:
                time.sleep(0.01)
                continue

            # Convert BGR to RGB for inference
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            annotated_rgb, status, _, timing, fps = detect(
                rgb,
                confidence=0.30,
                model_key=current_key,
                imgsz=imgsz,
                filter_pedestrians=filter_pedestrians,
                save_violations=save_violations,
            )

            # Convert back to BGR for OpenCV display
            display = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

            # Draw Real-Time HUD Overlay
            hud_bg = (20, 24, 30)
            cv2.rectangle(display, (10, 10), (450, 85), hud_bg, -1)
            cv2.rectangle(display, (10, 10), (450, 85), (60, 70, 85), 1)

            # Status and metrics
            status_color = (0, 0, 255) if 'WARNING' in status or '🚨' in status else (0, 255, 120)
            cv2.putText(display, f"FPS: {fps:.1f} | {current_key.upper()} ({imgsz}px)",
                        (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(display, status, (20, 54), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, status_color, 1, cv2.LINE_AA)
            ped_status = "ON (Riders Only)" if filter_pedestrians else "OFF (All Heads)"
            cv2.putText(display, f"Pedestrian Filter: {ped_status} [P]",
                        (20, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 200, 220), 1, cv2.LINE_AA)

            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('p'):
                filter_pedestrians = not filter_pedestrians
                print(f"Pedestrian filter toggled: {filter_pedestrians}")
            elif key == ord('m'):
                current_key = "medium" if current_key == "nano" and "medium" in AVAILABLE_MODELS else "nano"
                print(f"Switched model to: {current_key}")
            elif key == ord('s'):
                save_violation_evidence(display, 999)
    finally:
        cam.release()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Gradio Web Interface Helpers
# ---------------------------------------------------------------------------
def predict_photo(image, confidence, model_choice, imgsz, filter_pedestrians):
    if image is None:
        raise gr.Error('Please upload an image or take a photograph first.')
    annotated, status, rows, timing, _ = detect(
        image,
        confidence=confidence,
        model_key=model_choice,
        imgsz=int(imgsz),
        filter_pedestrians=filter_pedestrians,
        save_violations=False,
    )
    return annotated, status, rows, timing


def stream_webcam(image, confidence, model_choice, filter_pedestrians):
    if image is None:
        return None, "", ""
    annotated, status, _, timing, _ = detect(
        image,
        confidence=confidence,
        model_key=model_choice,
        imgsz=320,  # Fast 320px for smooth browser webcam streaming
        filter_pedestrians=filter_pedestrians,
        save_violations=False,
    )
    return annotated, status, timing


def process_video_file(video_path, confidence, model_choice, imgsz, filter_pedestrians,
                       save_violations, progress=gr.Progress()):
    """Analyze a recorded video file (.mp4) frame by frame and return an annotated video."""
    if not video_path:
        raise gr.Error("Please upload a video file (.mp4, .avi, .mov).")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error("Failed to open video file.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = os.path.join(os.path.dirname(video_path), "processed_" + os.path.basename(video_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    total_helmets = 0
    total_violations = 0

    progress(0.05, desc="Analyzing video frames...")
    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Run detection
        annotated_bgr, status, rows, _, _ = detect(
            frame_bgr,
            confidence=confidence,
            model_key=model_choice,
            imgsz=int(imgsz),
            filter_pedestrians=filter_pedestrians,
            save_violations=save_violations,
        )

        # Count statistics
        for row in rows:
            if row[0] == 'Helmet':
                total_helmets += 1
            elif row[0] == 'No Helmet':
                total_violations += 1

        out.write(annotated_bgr)
        if frame_idx % 10 == 0 and total_frames > 0:
            progress(frame_idx / total_frames, desc=f"Processed {frame_idx}/{total_frames} frames...")

    cap.release()
    out.release()
    summary = (f"Analysis Complete: Processed {frame_idx} frames.\n"
               f"Total Helmet Detections: {total_helmets}\n"
               f"Total No-Helmet Violations: {total_violations}")
    return output_path, summary


def get_example_images():
    if not os.path.isdir('images'):
        return []
    return [os.path.join('images', name) for name in sorted(os.listdir('images'))
            if name.lower().endswith(('.png', '.jpg', '.jpeg'))]


# ---------------------------------------------------------------------------
# Gradio User Interface
# ---------------------------------------------------------------------------
CSS = """
.gradio-container {max-width:1200px !important; margin:auto; font-family:Arial,sans-serif;}
#hero {padding:24px 28px; border-radius:18px; background:linear-gradient(120deg,#0f172a,#164e63); color:white; margin-bottom:16px;}
#hero h1 {font-size:28px; line-height:1.2; margin:6px 0 10px; color:white;}
#hero p {color:#dbeafe; max-width:800px; margin:0; line-height:1.5;}
.eyebrow {font-size:12px; letter-spacing:2px; font-weight:700; color:#67e8f9;}
.legend {display:flex; gap:20px; flex-wrap:wrap; padding:10px 0 14px; font-size:14px;}
.legend span {display:inline-flex; align-items:center; gap:8px;}
.dot {height:10px; width:10px; border-radius:50%; display:inline-block;}
.green {background:#16a34a;} .red {background:#dc2626;} .gray {background:#78909c;}
"""


def build_interface():
    theme = gr.themes.Soft(primary_hue='teal', neutral_hue='slate', font=['Arial', 'sans-serif'])
    style = {'theme': theme, 'css': CSS}
    major = int(gr.__version__.split('.')[0])

    model_choices = [desc for _, (_, desc) in AVAILABLE_MODELS.items()]
    photo_model_default = AVAILABLE_MODELS["medium"][1] if "medium" in AVAILABLE_MODELS else list(AVAILABLE_MODELS.values())[0][1]
    cam_model_default = AVAILABLE_MODELS["nano"][1] if "nano" in AVAILABLE_MODELS else photo_model_default

    with gr.Blocks(title='Automated Helmet Detection System', **(style if major < 6 else {})) as interface:
        gr.HTML('''<div id="hero">
            <div class="eyebrow">INTELLIGENT TRAFFIC ENFORCEMENT</div>
            <h1>Automated Motorcycle Helmet Detection</h1>
            <p>Real-time computer vision system to detect motorcycle riders and enforce helmet compliance.
            Includes high-speed CPU optimizations and intelligent pedestrian suppression.</p>
        </div>''')

        gr.HTML('''<div class="legend">
            <span><i class="dot green"></i>Helmet (Compliant)</span>
            <span><i class="dot red"></i>No Helmet (Violation)</span>
            <span><i class="dot gray"></i>Pedestrian (Suppressed / No Alert)</span>
        </div>''')

        with gr.Tabs():
            # ----------------- TAB 1: PHOTO CHECK -----------------
            with gr.Tab('📷 Photo Check'):
                with gr.Row():
                    with gr.Column(scale=1):
                        photo = gr.Image(sources=['upload', 'webcam'], type='pil', label='Upload Image or Take Photo', height=360)
                        with gr.Row():
                            photo_model = gr.Dropdown(choices=model_choices, value=photo_model_default, label='Model Selection')
                            photo_res = gr.Dropdown(choices=['320', '480', '640'], value='640', label='Resolution (imgsz)')
                        photo_thresh = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label='Confidence Threshold')
                        photo_ped = gr.Checkbox(value=False, label='Filter Pedestrians (Motorcycle Riders Only)')
                        with gr.Row():
                            run_btn = gr.Button('Detect Helmets', variant='primary')
                            clear_btn = gr.Button('Clear')
                    with gr.Column(scale=1):
                        result_img = gr.Image(type='pil', label='Detection Result', interactive=False, height=360)
                        status_box = gr.Textbox(label='Result Summary', interactive=False)
                        timing_box = gr.Textbox(label='Performance & Latency', interactive=False)
                photo_table = gr.Dataframe(headers=['Detection', 'Confidence (%)'], datatype=['str', 'number'],
                                           interactive=False, label='Detected Entities')
                run_btn.click(
                    predict_photo,
                    [photo, photo_thresh, photo_model, photo_res, photo_ped],
                    [result_img, status_box, photo_table, timing_box],
                )
                clear_btn.click(lambda: (None, None, '', [], ''), outputs=[photo, result_img, status_box, photo_table, timing_box])
                examples = get_example_images()
                if examples:
                    gr.Examples(examples=examples, inputs=photo)

            # ----------------- TAB 2: LIVE WEBCAM -----------------
            with gr.Tab('📹 Live Webcam (Browser)'):
                gr.Markdown('**Note for testing at your desk:** Keep **Filter Pedestrians UNCHECKED** when testing indoors without a bike. '
                            'Check it only when monitoring street traffic with real motorcycles. '
                            'For ultra-smooth 30+ FPS, run `python app.py --webcam` in your terminal!')
                with gr.Row():
                    with gr.Column():
                        webcam = gr.Image(sources=['webcam'], streaming=True, type='pil', label='Browser Webcam', height=360)
                        with gr.Row():
                            cam_model = gr.Dropdown(choices=model_choices, value=photo_model_default, label='Model Selection')
                            cam_ped = gr.Checkbox(value=False, label='Filter Pedestrians (Requires Motorcycle)')
                        cam_thresh = gr.Slider(0.10, 0.90, value=0.25, step=0.05, label='Confidence')
                    with gr.Column():
                        live_out = gr.Image(type='pil', interactive=False, label='Live Detections', height=360)
                        live_status = gr.Textbox(label='Live Status', interactive=False)
                        live_fps = gr.Textbox(label='Live Speed & Latency', interactive=False)

                stream_opts = {'concurrency_limit': 1, 'concurrency_id': 'helmet-stream', 'show_progress': 'hidden'}
                sig = inspect.signature(webcam.stream).parameters
                if 'stream_every' in sig:
                    stream_opts['stream_every'] = 0.2
                webcam.stream(stream_webcam, [webcam, cam_thresh, cam_model, cam_ped],
                              [live_out, live_status, live_fps], **stream_opts)

            # ----------------- TAB 3: VIDEO FILE ANALYSIS -----------------
            with gr.Tab('🎬 Video File Analysis (.mp4)'):
                gr.Markdown('Upload recorded traffic CCTV or dashcam footage. The system processes the video and exports an annotated copy.')
                with gr.Row():
                    with gr.Column():
                        video_input = gr.Video(label='Upload Traffic Video')
                        with gr.Row():
                            vid_model = gr.Dropdown(choices=model_choices, value=cam_model_default, label='Model')
                            vid_res = gr.Dropdown(choices=['320', '480', '640'], value='320', label='Resolution')
                        vid_thresh = gr.Slider(0.10, 0.90, value=0.30, step=0.05, label='Confidence')
                        with gr.Row():
                            vid_ped = gr.Checkbox(value=True, label='Filter Pedestrians')
                            vid_save = gr.Checkbox(value=True, label='Save Violation Snapshots')
                        vid_btn = gr.Button('Process Traffic Video', variant='primary')
                    with gr.Column():
                        video_output = gr.Video(label='Annotated Video')
                        vid_summary = gr.Textbox(label='Analysis Report', interactive=False)

                vid_btn.click(
                    process_video_file,
                    [video_input, vid_thresh, vid_model, vid_res, vid_ped, vid_save],
                    [video_output, vid_summary],
                )

        gr.HTML('<p style="font-size:12px; color:#64748b; margin-top:20px;">'
                'Automated Helmet Detection System | Optimized for real-time edge computing.</p>')

    return interface, (style if major >= 6 else {})


# ---------------------------------------------------------------------------
# Main Execution Entrypoint
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Automated Motorcycle Helmet Detection')
    parser.add_argument('--webcam', action='store_true', help='Launch direct zero-lag OpenCV webcam')
    parser.add_argument('--camera', type=int, default=0, help='Webcam index (default 0)')
    parser.add_argument('--model', type=str, default='nano', choices=['nano', 'medium'],
                        help="Model to use: 'nano' (YOLOv11 fast) or 'medium' (YOLOv8 precise)")
    parser.add_argument('--imgsz', type=int, default=320, choices=[320, 480, 640],
                        help='Inference resolution (default 320 for real-time speed)')
    parser.add_argument('--filter-pedestrians', action='store_true', default=True,
                        help='Suppress detections on pedestrians; only flag motorcycle riders')
    parser.add_argument('--port', type=int, default=7860, help='Gradio web server port')
    args = parser.parse_args()

    if args.webcam:
        run_opencv_camera(
            camera_index=args.camera,
            model_key=args.model,
            imgsz=args.imgsz,
            filter_pedestrians=args.filter_pedestrians,
        )
    else:
        interface, launch_style = build_interface()
        interface.queue(max_size=8).launch(server_port=args.port, **launch_style)
