"""FastAPI web application for wind turbine detection.

Serves single-image and bulk ZIP upload endpoints, running a YOLOv8n model to
detect wind turbines with confidence scores and configurable low-confidence flagging.
"""
import base64
import csv
import io
import os
import zipfile
from typing import Any

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

MODEL_PATH = os.environ.get("MODEL_PATH", "model/best.pt")
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.35"))

detection_model = YOLO(MODEL_PATH)

app = FastAPI(title="Wind Turbine Detection API")
last_bulk_results: list[dict[str, Any]] = []


def merge_stacked_boxes(
    boxes: list[list[float]],
    confs: list[float],
    x_overlap_thresh: float = 0.5,
    y_gap_thresh: float = 30,
) -> list[tuple[list[float], float]]:
    """Merge boxes likely belonging to the same physical structure.

    Sub-parts of one turbine (e.g. blades and tower) can be detected as separate
    boxes; this merges boxes that overlap significantly on the x-axis and sit
    close together vertically into a single detection, avoiding double-counting.

    Args:
        boxes: List of [x0, y0, x1, y1] bounding boxes.
        confs: Confidence score for each box, same order as boxes.
        x_overlap_thresh: Minimum horizontal overlap ratio to consider merging.
        y_gap_thresh: Maximum vertical gap (pixels) to consider merging.

    Returns:
        List of (merged_box, best_confidence) tuples.
    """
    items = sorted(zip(boxes, confs), key=lambda b: b[0][1])
    used = [False] * len(items)
    merged: list[tuple[list[float], float]] = []

    for i in range(len(items)):
        if used[i]:
            continue
        x0, y0, x1, y1 = items[i][0]
        best_conf = items[i][1]
        used[i] = True

        changed = True
        while changed:
            changed = False
            for j in range(len(items)):
                if used[j]:
                    continue
                jx0, jy0, jx1, jy1 = items[j][0]
                x_overlap = max(0, min(x1, jx1) - max(x0, jx0))
                x_overlap_ratio = x_overlap / max(1, min(x1 - x0, jx1 - jx0))
                y_gap = max(0, jy0 - y1, y0 - jy1)
                if x_overlap_ratio > x_overlap_thresh and y_gap < y_gap_thresh:
                    x0, y0, x1, y1 = min(x0, jx0), min(y0, jy0), max(x1, jx1), max(y1, jy1)
                    best_conf = max(best_conf, items[j][1])
                    used[j] = True
                    changed = True

        merged.append(([x0, y0, x1, y1], best_conf))

    return merged


def run_detection(image: Image.Image) -> list[tuple[list[float], float]]:
    """Run the detection model on an image and merge sub-part boxes.

    Args:
        image: A PIL Image to run detection on.

    Returns:
        List of (merged_box, confidence) tuples.
    """
    results = detection_model.predict(image, conf=0.10)
    raw_boxes = results[0].boxes.xyxy.tolist()
    raw_confs = results[0].boxes.conf.tolist()
    return merge_stacked_boxes(raw_boxes, raw_confs)


def draw_annotations(
    image: Image.Image,
    merged_detections: list[tuple[list[float], float]],
    confidence_threshold: float,
) -> Image.Image:
    """Draw bounding boxes and confidence labels on a copy of the image.

    Labels are staggered vertically when boxes are close together to avoid
    overlapping text, and sizing scales with image resolution for legibility.

    Args:
        image: Source PIL Image.
        merged_detections: Output of run_detection.
        confidence_threshold: Threshold below which a detection is flagged.

    Returns:
        A new PIL Image with annotations drawn.
    """
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    box_color = (37, 99, 235)
    img_scale = max(annotated.width, annotated.height) / 500
    line_width = max(5, int(6 * img_scale))
    font_size = max(24, int(28 * img_scale))

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size
        )
    except OSError:
        font = ImageFont.load_default()

    ordered = sorted(merged_detections, key=lambda d: d[0][0])
    placed_label_boxes: list[tuple[float, float, float, float]] = []

    for box, conf in ordered:
        label = f"wind_turbine {conf * 100:.1f}%"
        draw.rectangle(box, outline=box_color, width=line_width)

        text_bbox = draw.textbbox((0, 0), label, font=font)
        label_w = text_bbox[2] - text_bbox[0] + 14
        label_h = text_bbox[3] - text_bbox[1] + 12
        label_x = box[0]
        label_y = max(0, box[1] - label_h - 3)

        while any(
            not (
                label_x + label_w < px
                or label_x > px + pw
                or label_y + label_h < py
                or label_y > py + ph
            )
            for (px, py, pw, ph) in placed_label_boxes
        ):
            label_y += label_h + 3

        draw.rectangle([label_x, label_y, label_x + label_w, label_y + label_h], fill=box_color)
        draw.text((label_x + 7, label_y + 6), label, fill=(255, 255, 255), font=font)
        placed_label_boxes.append((label_x, label_y, label_w, label_h))

    return annotated


@app.get("/")
def read_root() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "Wind Turbine Detection API is running"}


@app.post("/detect")
async def detect_single_image(
    file: UploadFile = File(...),
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """Detect wind turbines in a single uploaded image.

    Args:
        file: The uploaded image file.
        confidence_threshold: Detections below this are flagged for review.

    Returns:
        Detection results including an annotated image (base64-encoded JPEG).
    """
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    merged = run_detection(image)

    detections = [
        {
            "bbox": {"x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3]},
            "confidence": round(conf * 100, 1),
            "needs_review": conf < confidence_threshold,
        }
        for box, conf in merged
    ]

    annotated_image = draw_annotations(image, merged, confidence_threshold)
    buf = io.BytesIO()
    annotated_image.save(buf, format="JPEG")
    annotated_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "filename": file.filename,
        "turbine_count": len(detections),
        "detections": detections,
        "annotated_image_base64": annotated_b64,
        "message": "No turbines detected" if len(detections) == 0 else None,
        "confidence_threshold_used": confidence_threshold,
    }


@app.post("/detect-bulk")
async def detect_bulk_zip(
    file: UploadFile = File(...),
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """Detect wind turbines in every image inside an uploaded ZIP file.

    Args:
        file: The uploaded ZIP file containing images.
        confidence_threshold: Detections below this are flagged for review.

    Returns:
        Per-image detection summary for the whole batch.
    """
    global last_bulk_results
    zip_bytes = await file.read()
    results_summary = []

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        image_names = [
            n for n in z.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        for img_name in image_names:
            with z.open(img_name) as img_file:
                image = Image.open(img_file).convert("RGB")

            merged = run_detection(image)
            confidences = [round(conf * 100, 1) for _, conf in merged]
            needs_review = any(c < confidence_threshold * 100 for c in confidences)

            results_summary.append(
                {
                    "filename": img_name,
                    "turbine_count": len(confidences),
                    "confidences": confidences,
                    "needs_review": needs_review,
                }
            )

    last_bulk_results = results_summary
    return {
        "total_images": len(results_summary),
        "results": results_summary,
        "confidence_threshold_used": confidence_threshold,
    }


@app.get("/detect-bulk/export-csv")
def export_bulk_csv() -> StreamingResponse:
    """Export the most recent bulk detection results as a downloadable CSV."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Filename", "Turbine Count", "Confidences (%)", "Needs Review"])
    for row in last_bulk_results:
        writer.writerow(
            [
                row["filename"],
                row["turbine_count"],
                "; ".join(str(c) for c in row["confidences"]),
                "YES" if row["needs_review"] else "No",
            ]
        )
    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=bulk_results.csv"},
    )


@app.get("/app", response_class=HTMLResponse)
def serve_frontend() -> str:
    """Serve the single-page frontend for uploading images and viewing results."""
    return """
    <!DOCTYPE html><html><head><title>Wind Turbine Detection</title>
    <style>
    body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; }
    h1 { color: #2c3e50; } #result-img { max-width: 100%; margin-top: 20px; border: 1px solid #ddd; }
    table { width: 100%; border-collapse: collapse; margin-top: 15px; }
    th, td { padding: 8px; border: 1px solid #ddd; text-align: left; } th { background: #f4f4f4; }
    .flagged { background: #fff3cd; color: #856404; font-weight: bold; }
    #status, #bulk-status { margin-top: 15px; font-weight: bold; }
    .tabs { margin-bottom: 20px; } .tab-btn { padding: 10px 20px; cursor: pointer; border: 1px solid #ccc; background: #f4f4f4; }
    .tab-btn.active { background: #2c3e50; color: white; } .tab-content { display: none; } .tab-content.active { display: block; }
    </style></head><body>
    <h1>Wind Turbine Detection</h1>
    <div class="tabs"><button class="tab-btn active" onclick="showTab('single')">Single Image</button><button class="tab-btn" onclick="showTab('bulk')">Bulk ZIP Upload</button></div>
    <div id="single-tab" class="tab-content active">
    <p>Upload an image to detect wind turbines with confidence scores.</p>
    <input type="file" id="imageInput" accept="image/*"><button onclick="detectTurbines()">Detect</button>
    <div id="status"></div><img id="result-img" style="display:none;">
    <table id="results-table" style="display:none;"><thead><tr><th>#</th><th>Confidence</th><th>Status</th></tr></thead><tbody id="results-body"></tbody></table>
    </div>
    <div id="bulk-tab" class="tab-content">
    <p>Upload a ZIP of images for batch turbine detection.</p>
    <input type="file" id="zipInput" accept=".zip"><button onclick="detectBulk()">Detect All</button>
    <div id="bulk-status"></div>
    <table id="bulk-results-table" style="display:none;"><thead><tr><th>Filename</th><th>Turbines Found</th><th>Confidences</th><th>Status</th></tr></thead><tbody id="bulk-results-body"></tbody></table>
    <button id="csv-btn" style="display:none; margin-top:15px;" onclick="downloadCsv()">Download CSV</button>
    </div>
    <script>
    function showTab(tab) {
        document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
        document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
        document.getElementById(tab + '-tab').classList.add('active');
        event.target.classList.add('active');
    }
    async function detectTurbines() {
        const fileInput = document.getElementById('imageInput');
        if (!fileInput.files.length) { alert('Please select an image first'); return; }
        const formData = new FormData(); formData.append('file', fileInput.files[0]);
        document.getElementById('status').innerText = 'Detecting...';
        const response = await fetch('/detect', { method: 'POST', body: formData });
        const data = await response.json();
        document.getElementById('status').innerText = data.message || `Found ${data.turbine_count} turbine(s)`;
        const img = document.getElementById('result-img');
        img.src = 'data:image/jpeg;base64,' + data.annotated_image_base64; img.style.display = 'block';
        const tbody = document.getElementById('results-body'); tbody.innerHTML = '';
        data.detections.forEach((d, i) => {
            const row = document.createElement('tr'); if (d.needs_review) row.className = 'flagged';
            row.innerHTML = `<td>${i+1}</td><td>${d.confidence}%</td><td>${d.needs_review ? 'Needs Review' : 'OK'}</td>`;
            tbody.appendChild(row);
        });
        document.getElementById('results-table').style.display = data.detections.length ? 'table' : 'none';
    }
    async function detectBulk() {
        const fileInput = document.getElementById('zipInput');
        if (!fileInput.files.length) { alert('Please select a ZIP file first'); return; }
        const formData = new FormData(); formData.append('file', fileInput.files[0]);
        document.getElementById('bulk-status').innerText = 'Processing ZIP...';
        const response = await fetch('/detect-bulk', { method: 'POST', body: formData });
        const data = await response.json();
        document.getElementById('bulk-status').innerText = `Processed ${data.total_images} image(s)`;
        const tbody = document.getElementById('bulk-results-body'); tbody.innerHTML = '';
        data.results.forEach(r => {
            const row = document.createElement('tr'); if (r.needs_review) row.className = 'flagged';
            row.innerHTML = `<td>${r.filename}</td><td>${r.turbine_count}</td><td>${r.confidences.join(', ')}%</td><td>${r.needs_review ? 'Review' : 'OK'}</td>`;
            tbody.appendChild(row);
        });
        document.getElementById('bulk-results-table').style.display = 'table';
        document.getElementById('csv-btn').style.display = 'inline-block';
    }
    function downloadCsv() { window.location.href = '/detect-bulk/export-csv'; }
    </script></body></html>
    """