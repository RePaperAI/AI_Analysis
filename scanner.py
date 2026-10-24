"""Small Flask + OpenCV service used by the RePaper AI dashboard.

The scanner is intentionally explainable: it finds the sheet, flattens it,
then uses grayscale/adaptive thresholding to estimate ink and large damaged
regions. It is a useful first pass, not an OCR or forensic paper-grading model.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(ROOT))


@app.after_request
def allow_local_dashboard(response):
    # Also supports opening dashboard.html directly from disk (file://).
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


def _order_points(points: np.ndarray) -> np.ndarray:
    points = points.reshape(4, 2).astype("float32")
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).ravel()
    return np.array([points[np.argmin(sums)], points[np.argmin(diffs)],
                     points[np.argmax(sums)], points[np.argmax(diffs)]], dtype="float32")


def _flatten_sheet(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Return a top-down sheet image and a confidence score for its detection."""
    height, width = image.shape[:2]
    scale = min(1.0, 1200.0 / max(height, width))
    small = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image.copy()
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    image_area = small.shape[0] * small.shape[1]
    for contour in contours[:25]:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True) if perimeter else []
        area = cv2.contourArea(contour)
        if len(approx) == 4 and area > image_area * 0.25:
            quad = _order_points(approx)
            w = int(max(np.linalg.norm(quad[2] - quad[3]), np.linalg.norm(quad[1] - quad[0])))
            h = int(max(np.linalg.norm(quad[1] - quad[2]), np.linalg.norm(quad[0] - quad[3])))
            if w < 100 or h < 100:
                break
            target_w, target_h = 900, max(1100, int(900 * h / max(w, 1)))
            destination = np.array([[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]], dtype="float32")
            matrix = cv2.getPerspectiveTransform(quad, destination)
            return cv2.warpPerspective(small, matrix, (target_w, target_h)), min(99.0, 70 + area / image_area * 30)
    return small, 48.0


def _make_cutting_plan(sheet: np.ndarray, threshold: np.ndarray, demand: dict) -> tuple[str, list[dict], str]:
    """Pack rectangles only into connected, ink-free areas of the flattened sheet."""
    height, width = threshold.shape
    products = {
        # Product dimensions are the user-provided inch sizes converted to mm.
        "sticky": ("Sticky Notes", 76.2, 76.2, (54, 190, 74)),
        "memo": ("Memo Sheets", 76.2, 127.0, (44, 157, 211)),
        "flashcards": ("Flashcards", 76.2, 127.0, (153, 76, 194)),
        "scratch": ("Scratch Sheets", 101.6, 152.4, (220, 157, 0)),
    }
    # Expand ink by a safety margin so cut lines do not run through nearby text,
    # handwriting, borders, or illustrations.
    ink = (threshold > 0).astype(np.uint8)
    ink[[0, 1, -2, -1], :] = 1
    ink[:, [0, 1, -2, -1]] = 1
    ink = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    blank = (ink == 0).astype(np.uint8)
    integral = cv2.integral(blank)
    reserved = np.zeros_like(blank)
    pieces = []
    margin_x, margin_y = int(width * .05), int(height * .05)

    def blank_ratio(x: int, y: int, w: int, h: int) -> float:
        total = integral[y + h, x + w] - integral[y, x + w] - integral[y + h, x] + integral[y, x]
        return float(total) / max(1, w * h)

    def place_product(product_key: str, quota: int) -> None:
        label, mm_w, mm_h, color = products[product_key]
        px_per_mm = min(width / 216.0, height / 279.0)
        # Try both orientations so memo sheets and flashcards use space efficiently.
        orientations = [(mm_w, mm_h), (mm_h, mm_w)]
        for physical_w, physical_h in orientations:
            if len(pieces) >= 12 or sum(1 for p in pieces if p["product"] == product_key) >= quota:
                break
            piece_w, piece_h = max(32, int(physical_w * px_per_mm)), max(32, int(physical_h * px_per_mm))
            # Scan candidate origins at a fine stride. A candidate must be effectively
            # 100% blank; the dilation above supplies practical cutting clearance.
            for y in range(margin_y, height - margin_y - piece_h + 1, 6):
                for x in range(margin_x, width - margin_x - piece_w + 1, 6):
                    if len(pieces) >= 12 or sum(1 for p in pieces if p["product"] == product_key) >= quota:
                        break
                    if np.any(reserved[y:y + piece_h, x:x + piece_w]):
                        continue
                    if blank_ratio(x, y, piece_w, piece_h) >= .995:
                        reserved[y:y + piece_h, x:x + piece_w] = 1
                        pieces.append({"x": x, "y": y, "width": piece_w, "height": piece_h,
                                       "label": label, "product": product_key,
                                       "size_mm": f"{physical_w} x {physical_h} mm"})

    demand_values = {key: max(0.0, float(demand.get(key, 0))) for key in products}
    if not any(demand_values.values()):
        demand_values["sticky"] = 1
    max_demand = max(demand_values.values())
    # High-demand items get up to three requested pieces; lower-demand items get
    # fewer. Every non-zero demand category gets at least one attempt.
    for product_key, value in sorted(demand_values.items(), key=lambda item: item[1], reverse=True):
        if value <= 0:
            continue
        quota = max(1, int(round(3 * value / max_demand)))
        place_product(product_key, quota)

    annotated = sheet.copy()
    for piece in pieces:
        x, y, w, h = piece["x"], piece["y"], piece["width"], piece["height"]
        piece_color = products[piece["product"]][3]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), piece_color, 5)
        cv2.rectangle(annotated, (x, y), (x + min(w, 190), y + 34), (255, 255, 255), -1)
        cv2.putText(annotated, piece["label"], (x + 8, y + 24), cv2.FONT_HERSHEY_SIMPLEX, .7, piece_color, 2, cv2.LINE_AA)
    if not pieces:
        status = "No clean product-sized region found; recycle recommended"
    else:
        counts = {}
        for piece in pieces:
            counts[piece["label"]] = counts.get(piece["label"], 0) + 1
        summary = ", ".join(f"{count} {name}" for name, count in counts.items())
        status = f"Planned: {summary}; remaining marked content goes to recycling"
    ok, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        return "", pieces, status
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii"), pieces, status


def analyze_sheet(image: np.ndarray, demand: dict | None = None) -> dict:
    sheet, detection_confidence = _flatten_sheet(image)
    gray = cv2.cvtColor(sheet, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    margin_y, margin_x = int(height * 0.04), int(width * 0.04)
    roi = gray[margin_y:height - margin_y, margin_x:width - margin_x]

    # Adaptive threshold handles shadows and uneven lighting better than a fixed threshold.
    threshold = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 31, 14)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    ink_pixels = int(np.count_nonzero(threshold))
    total_pixels = threshold.size

    # Very large, dense connected components are treated as damage. Normal text strokes
    # remain in the printed estimate, while a stain/tear/black block is separated out.
    components, labels, stats, _ = cv2.connectedComponentsWithStats(threshold, 8)
    damage_mask = np.zeros_like(threshold)
    damage_limit = total_pixels * 0.012
    for idx in range(1, components):
        area = stats[idx, cv2.CC_STAT_AREA]
        component_w = stats[idx, cv2.CC_STAT_WIDTH]
        component_h = stats[idx, cv2.CC_STAT_HEIGHT]
        if area > damage_limit and component_w > width * 0.08 and component_h > height * 0.02:
            damage_mask[labels == idx] = 255
    damage_pixels = int(np.count_nonzero(damage_mask))
    printed_pixels = max(0, ink_pixels - damage_pixels)

    printed = round(printed_pixels / total_pixels * 100, 1)
    damage = round(damage_pixels / total_pixels * 100, 1)
    usable = round(max(0.0, 100 - printed - damage), 1)
    # Confidence falls when the image is extremely dark or the sheet boundary is unclear.
    lighting = float(np.mean(roi))
    quality = 100 - min(45, abs(lighting - 190) / 4)
    confidence = round(max(50, min(99, detection_confidence * 0.65 + quality * 0.35)))
    full_ink = np.zeros_like(gray, dtype=np.uint8)
    full_ink[margin_y:height - margin_y, margin_x:width - margin_x] = threshold
    plan_image, pieces, plan_status = _make_cutting_plan(sheet, full_ink, demand or {})
    return {
        "blank": usable,
        "printed": printed,
        "damage": damage,
        "confidence": confidence,
        "type": "document detected" if detection_confidence > 60 else "best-effort scan",
        "width": int(width),
        "height": int(height),
        "plan_image": plan_image,
        "pieces": pieces,
        "plan_status": plan_status,
    }


@app.get("/")
def dashboard():
    return send_from_directory(ROOT, "dashboard.html")


@app.get("/<path:filename>")
def static_files(filename: str):
    """Serve the dashboard's local JS, CSS, and image assets."""
    return send_from_directory(ROOT, filename)


@app.post("/api/analyze")
def analyze():
    uploaded = request.files.get("image")
    if not uploaded or not uploaded.filename:
        return jsonify(error="Upload an image of a sheet of paper."), 400
    raw = uploaded.read()
    if len(raw) > 10 * 1024 * 1024:
        return jsonify(error="Image must be smaller than 10 MB."), 413
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return jsonify(error="OpenCV could not read that image. Use JPG, PNG, or WebP."), 400
    try:
        try:
            demand = json.loads(request.form.get("demand", "{}"))
        except json.JSONDecodeError:
            demand = {}
        return jsonify(analyze_sheet(image, demand))
    except Exception as exc:
        app.logger.exception("Image analysis failed")
        return jsonify(error=f"Image analysis failed: {exc}"), 422


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
