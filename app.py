"""Motorcycle helmet detection.

Run from the folder containing this file and your existing weights:
    python app.py                 # browser: photographs and live webcam
    python app.py --webcam        # direct OpenCV webcam window (local PC)
    python app.py --webcam --camera 1   # optional external camera

The original model-loading block and weight filenames are unchanged.
No training or weight modification takes place in this application.
"""

import os
import torch

# Original cache configuration retained.
os.environ['TORCH_HOME'] = '/tmp/torch_cache'
os.environ['HF_HOME'] = '/tmp/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'
os.environ['TMPDIR'] = '/tmp'
torch.hub.set_dir('/tmp/torch_hub')

import argparse
import inspect
import threading
import time

import cv2
import gradio as gr
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageOps

# Load models with priority to YOLOv8
# Try to load YOLOv8 model first, fall back to YOLOv11 if not available
model = None
model_name = ""

if os.path.exists("best.pt"):
    model = YOLO("best.pt")
    model_name = "YOLOv8 (best.pt)"
    print("✓ Loaded YOLOv8 model (best.pt)")
elif os.path.exists("yolov11nbest.pt"):
    model = YOLO("yolov11nbest.pt")
    model_name = "YOLOv11 (yolov11nbest.pt)"
    print("✓ Loaded YOLOv11 model (yolov11nbest.pt)")
else:
    raise FileNotFoundError("No model file found. Please ensure 'best.pt' or 'yolov11nbest.pt' exists.")

# Names, not numeric IDs, are used because the backup has 13 classes.
HELMET_NAMES = {
    'accepthelmet', 'helmet', 'cyclinghelmet', 'halfface', 'hardhat',
    'modularhelmet', 'nutshell', 'quarterfacehelmet', 'sportshelmet',
}
NO_HELMET_NAMES = {'nonhelmet', 'nohelmet', 'withouthelmet', 'nothelmet'}
INFERENCE_LOCK = threading.Lock()


def display_class(raw_name):
    key = ''.join(c for c in str(raw_name).lower() if c.isalnum())
    if key in NO_HELMET_NAMES:
        return 'No Helmet', (220, 38, 38)
    if key in HELMET_NAMES:
        return 'Helmet', (22, 163, 74)
    # Preserve other backup classes; never infer No Helmet from a person box.
    return str(raw_name), (37, 99, 235)


def draw_detection(canvas, xyxy, text, colour):
    """Draw directly on an RGB array, preserving the image's original colours."""
    height, width = canvas.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy]
    x1, x2 = sorted((max(0, min(x1, width-1)), max(0, min(x2, width-1))))
    y1, y2 = sorted((max(0, min(y1, height-1)), max(0, min(y2, height-1))))
    thickness = max(1, min(4, round(min(width, height) / 300)))
    scale = max(0.4, min(0.9, min(width, height) / 850))
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, 1)
    if tw + 12 > width:
        scale *= max(1, width-12) / (tw+1)
        (tw, th), baseline = cv2.getTextSize(text, font, scale, 1)
    left = max(0, min(x1, width-tw-12))
    label_height = th + baseline + 10
    top = max(0, min(y1-label_height if y1 >= label_height else y1,
                     height-label_height))
    cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, thickness)
    cv2.rectangle(canvas, (left, top), (min(width-1, left+tw+12),
                  min(height-1, top+label_height)), colour, -1)
    cv2.putText(canvas, text, (left+6, top+th+4), font, scale,
                (255, 255, 255), 1, cv2.LINE_AA)


def detect(image, confidence=0.25):
    """Shared inference for uploaded images, browser video and OpenCV video."""
    if image is None:
        return None, 'Add a photograph or start the camera.', [], ''
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    image = ImageOps.exif_transpose(image).convert('RGB')
    start = time.perf_counter()
    # Only one request uses the shared predictor at a time.
    with INFERENCE_LOCK:
        result = model(image, conf=float(confidence), iou=0.7, verbose=False)[0]
        entries = [] if result.boxes is None else result.boxes.data.cpu().tolist()
        names = result.names
    canvas = np.asarray(image).copy()
    counts = {'Helmet': 0, 'No Helmet': 0}
    rows = []
    # Draw low-confidence boxes first so stronger detections remain readable.
    for entry in sorted(entries, key=lambda row: row[-2]):
        xyxy, score, class_id = entry[:4], float(entry[-2]), int(entry[-1])
        label, colour = display_class(names[class_id])
        if label in counts:
            counts[label] += 1
        rows.append([label, round(score*100, 1)])
        draw_detection(canvas, xyxy, f'{label} {score:.0%}', colour)
    rows.sort(key=lambda row: row[1], reverse=True)
    if counts['No Helmet']:
        status = (f"WARNING: No Helmet detected: {counts['No Helmet']} | "
                  f"Helmet detected: {counts['Helmet']}")
    elif counts['Helmet']:
        status = f"Helmet detected: {counts['Helmet']} | No Helmet detected: 0"
    else:
        status = 'No helmet-related detection. This does not confirm helmet use.'
    elapsed = time.perf_counter()-start
    timing = f'Processing time: {elapsed:.2f} seconds per image/frame'
    return Image.fromarray(canvas), status, rows, timing


def predict(image, confidence, progress=gr.Progress()):
    if image is None:
        raise gr.Error('Please upload an image or take a photograph first.')
    progress(0.1, desc='Checking image…')
    output = detect(image, confidence)
    progress(1, desc='Detection complete')
    return output


def stream_frame(image, confidence):
    annotated, status, _, timing = detect(image, confidence)
    return annotated, status, timing


def get_example_images():
    if not os.path.isdir('images'):
        return []
    return [os.path.join('images', name) for name in sorted(os.listdir('images'))
            if name.lower().endswith(('.png', '.jpg', '.jpeg'))]


CSS = """
.gradio-container {max-width:1180px !important; margin:auto; font-family:Arial,sans-serif;}
#hero {padding:28px 32px; border-radius:20px; background:linear-gradient(120deg,#0f172a,#164e63); color:white; margin-bottom:18px;}
#hero h1 {font-size:30px; line-height:1.2; margin:8px 0 12px; color:white;}
#hero p {color:#dbeafe; max-width:760px; margin:0; line-height:1.6;}
.eyebrow {font-size:12px; letter-spacing:2px; font-weight:700; color:#67e8f9;}
.legend {display:flex; gap:20px; flex-wrap:wrap; padding:12px 0 16px; font-size:14px;}
.legend span {display:inline-flex; align-items:center; gap:8px;}
.dot {height:10px; width:10px; border-radius:50%; display:inline-block;}
.green {background:#16a34a;} .red {background:#dc2626;}
.footer-note {font-size:13px; opacity:.8; line-height:1.6; padding-top:12px;}
"""


def build_interface():
    # Gradio 6 moved theme/css to launch(); keep Gradio 4/5 compatibility.
    theme = gr.themes.Soft(primary_hue='teal', neutral_hue='slate',
                           font=['Arial', 'sans-serif'])
    style = {'theme': theme, 'css': CSS}
    major = int(gr.__version__.split('.')[0])
    with gr.Blocks(title='Motorcycle Helmet Detection',
                   **(style if major < 6 else {})) as interface:
        gr.HTML('''<div id="hero"><div class="eyebrow">MOTORCYCLE SAFETY</div>
            <h1>Helmet Detection</h1><p>Check a photograph or use your live camera
            to detect helmets and uncovered heads.</p></div>''')
        gr.Markdown(f'**Active model:** {model_name}')
        gr.HTML('''<div class="legend"><span><i class="dot green"></i>Helmet</span>
            <span><i class="dot red"></i>No Helmet</span>
            <span>Each box includes the detection confidence.</span></div>''')
        with gr.Tab('Photo check'):
            with gr.Row():
                with gr.Column():
                    photo = gr.Image(sources=['upload', 'webcam'], type='pil',
                                     label='Upload an image or take a photo', height=360)
                    threshold = gr.Slider(0.05, 0.95, value=0.25, step=0.05,
                                          label='Minimum confidence')
                    with gr.Row():
                        run = gr.Button('Detect helmets', variant='primary')
                        clear = gr.Button('Clear')
                with gr.Column():
                    result = gr.Image(type='pil', label='Detection result',
                                      interactive=False, height=360)
                    status = gr.Textbox(label='Result summary', interactive=False)
                    timing = gr.Textbox(label='Processing time', interactive=False)
            table = gr.Dataframe(headers=['Detection', 'Confidence (%)'],
                                 datatype=['str', 'number'], interactive=False,
                                 label='Detected objects')
            run.click(predict, [photo, threshold], [result, status, table, timing],
                      concurrency_limit=1, concurrency_id='helmet-model')
            clear.click(lambda: (None, None, '', [], ''), outputs=[photo, result, status, table, timing], queue=False)
            examples = get_example_images()
            if examples:
                gr.Examples(examples=examples, inputs=photo)
        with gr.Tab('Live webcam'):
            gr.Markdown('Allow camera access, then start the webcam stream using its controls. '
                        'Results update automatically; stop the stream to finish. '
                        'Speed depends on your computer. Each session runs for up to 5 minutes; '
                        'restart the stream to continue.')
            with gr.Row():
                camera = gr.Image(sources=['webcam'], streaming=True, type='pil',
                                  label='Live camera', height=360)
                live_result = gr.Image(type='pil', interactive=False,
                                       label='Live detections', height=360)
            live_threshold = gr.Slider(0.05, 0.95, value=0.25, step=0.05,
                                      label='Minimum confidence')
            live_status = gr.Textbox(label='Live result', interactive=False)
            live_time = gr.Textbox(label='Processing time', interactive=False)
            stream_options = {'concurrency_limit': 1, 'concurrency_id': 'helmet-model',
                              'show_progress': 'hidden'}
            supported = inspect.signature(camera.stream).parameters
            if 'stream_every' in supported:
                stream_options['stream_every'] = 0.3
            if 'time_limit' in supported:
                stream_options['time_limit'] = 300
            camera.stream(stream_frame, [camera, live_threshold],
                          [live_result, live_status, live_time], **stream_options)
        gr.HTML('<p class="footer-note">Confidence is the model’s score for a detection, '
                'not overall accuracy. No detection does not prove that a helmet is worn. '
                'Use clear lighting and keep the head visible.</p>')
    return interface, (style if major >= 6 else {})


def run_opencv_camera(camera_index=0):
    """Direct OpenCV capture on the computer running Python; not a Colab camera."""
    capture = cv2.VideoCapture(camera_index)
    window = 'Helmet Detection - press Q or Esc to stop'
    try:
        if not capture.isOpened():
            raise RuntimeError('Cannot open webcam. Close other camera apps and check camera permissions/index.')
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        while True:
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError('The webcam stopped providing frames. Check its connection.')
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            annotated, status, _, _ = detect(rgb, 0.25)
            display = cv2.cvtColor(np.asarray(annotated), cv2.COLOR_RGB2BGR)
            cv2.putText(display, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 0, 255) if status.startswith('WARNING') else (255, 255, 255),
                        1, cv2.LINE_AA)
            cv2.imshow(window, display)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Motorcycle helmet detection')
    parser.add_argument('--webcam', action='store_true', help='Use a local OpenCV camera window')
    parser.add_argument('--camera', type=int, default=0, help='OpenCV camera index (default 0)')
    args = parser.parse_args()
    if args.webcam:
        run_opencv_camera(args.camera)
    else:
        interface, launch_style = build_interface()
        interface.queue(max_size=8).launch(**launch_style)
