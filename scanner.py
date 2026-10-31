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
from cutting import CutterConfig, generate_gcode

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "pieces_config.json"
app = Flask(__name__, static_folder=str(ROOT))


def _load_products() -> dict:
    """Load configurable products and convert inch dimensions to millimetres."""
    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        raw = json.load(config_file)
    products = {}
    for key, item in raw["products"].items():
        products[key] = (
            item["label"],
            float(item["width_in"]) * 25.4,
            float(item["height_in"]) * 25.4,
            tuple(item["color"]),
        )
    return products


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


def _target_dimensions(width: float, height: float) -> tuple[int, int]:
    """Resize a flattened sheet without changing its aspect ratio."""
    scale = min(1.0, 1200.0 / max(1.0, width, height))
    return max(600, int(round(width * scale))), max(600, int(round(height * scale)))


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
            target_w, target_h = _target_dimensions(w, h)
            destination = np.array([[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]], dtype="float32")
            matrix = cv2.getPerspectiveTransform(quad, destination)
            return cv2.warpPerspective(small, matrix, (target_w, target_h)), min(99.0, 70 + area / image_area * 30)

    # Phone photos of blank paper often have no four strong Canny edges, and
    # the bottom edge may be outside the camera frame. In that case, locate the
    # largest bright paper region against the darker surroundings and use its
    # bounding rectangle as a best-effort sheet boundary.
    bright_cutoff = max(105, int(np.percentile(gray, 58)))
    bright = cv2.threshold(gray, bright_cutoff, 255, cv2.THRESH_BINARY)[1]
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31)))
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    bright_contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if bright_contours:
        contour = max(bright_contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area > image_area * 0.25:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 100 and h > 100:
                quad = np.array([[x, y], [x + w - 1, y],
                                 [x + w - 1, y + h - 1], [x, y + h - 1]], dtype="float32")
                target_w, target_h = _target_dimensions(w, h)
                destination = np.array([[0, 0], [target_w - 1, 0],
                                        [target_w - 1, target_h - 1], [0, target_h - 1]], dtype="float32")
                matrix = cv2.getPerspectiveTransform(quad, destination)
                return cv2.warpPerspective(small, matrix, (target_w, target_h)), min(85.0, 55 + area / image_area * 30)
    return small, 48.0


def _make_cutting_plan(sheet: np.ndarray, threshold: np.ndarray, demand: dict) -> tuple[str, list[dict], str]:
    """Pack rectangles only into connected, ink-free areas of the flattened sheet."""
    height, width = threshold.shape
    gray = cv2.cvtColor(sheet, cv2.COLOR_BGR2GRAY)
    products = _load_products()
    # Expand ink by a safety margin so cut lines do not run through nearby text,
    # handwriting, borders, or illustrations.
    ink = (threshold > 0).astype(np.uint8)
    ink[[0, 1, -2, -1], :] = 1
    ink[:, [0, 1, -2, -1]] = 1
    # The cutting mask uses a wider clearance than the layout preview so faint
    # exam text and thin lines cannot be mistaken for blank paper.
    ink = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)))
    blank = (ink == 0).astype(np.uint8)
    integral = cv2.integral(blank)
    # The detected border is already part of the ink mask, so keep only a small
    # outer search margin and preserve more usable paper near the edges.
    margin_x, margin_y = int(width * .02), int(height * .02)

    def blank_ratio(x: int, y: int, w: int, h: int) -> float:
        total = integral[y + h, x + w] - integral[y, x + w] - integral[y + h, x] + integral[y, x]
        return float(total) / max(1, w * h)

    demand_values = {key: max(0.0, float(demand.get(key, 0))) for key in products}
    if not any(demand_values.values()):
        demand_values["sticky"] = 1
    # Generate safe candidate rectangles for every demanded product and both
    # orientations. A candidate is accepted only when its complete interior is
    # blank after the ink safety dilation.
    candidates = []
    for product_key, demand_value in demand_values.items():
        if demand_value <= 0:
            continue
        label, mm_w, mm_h, _ = products[product_key]
        # A4 sheet coordinate system: 210 x 297 mm.
        px_per_mm = min(width / 210.0, height / 297.0)
        orientations = {(mm_w, mm_h), (mm_h, mm_w)}
        for physical_w, physical_h in orientations:
            piece_w, piece_h = max(32, int(physical_w * px_per_mm)), max(32, int(physical_h * px_per_mm))
            for y in range(margin_y, height - margin_y - piece_h + 1, 6):
                for x in range(margin_x, width - margin_x - piece_w + 1, 6):
                    ratio = blank_ratio(x, y, piece_w, piece_h)
                    # A cut rectangle must be completely clear after dilation;
                    # allowing even a small ink fraction can place mini pieces
                    # over text, barcodes, or answer boxes.
                    if ratio >= 0.99999:
                        area = piece_w * piece_h
                        # Demand is a preference, while area recovery is the main
                        # objective. This lets smaller products fill leftover gaps.
                        value = area * (0.45 + 0.55 * demand_value / 100.0)
                        candidates.append({"x": x, "y": y, "width": piece_w, "height": piece_h,
                                           "label": label, "product": product_key,
                                           "size_mm": f"{physical_w} x {physical_h} mm",
                                           "size_in": f"{physical_w / 25.4:g} x {physical_h / 25.4:g} in",
                                           "value": value})

    # Keep spatially diverse candidates so optimization remains fast and does not
    # overfit to hundreds of nearly identical origins in one blank region.
    diverse = {}
    for candidate in candidates:
        key = (candidate["product"], candidate["x"] // 48, candidate["y"] // 48,
               candidate["width"], candidate["height"])
        if key not in diverse or candidate["value"] > diverse[key]["value"]:
            diverse[key] = candidate
    # Preserve a useful candidate set for every product; otherwise a large
    # scratch/memo rectangle could crowd all smaller sticky-note candidates out
    # before the optimizer gets a chance to combine them.
    grouped = {}
    for candidate in diverse.values():
        grouped.setdefault(candidate["product"], []).append(candidate)
    candidates = []
    for group in grouped.values():
        candidates.extend(sorted(group, key=lambda item: item["value"], reverse=True)[:250])
    candidates = sorted(candidates, key=lambda item: item["value"], reverse=True)[:900]

    def overlaps(a: dict, b: dict) -> bool:
        return not (a["x"] + a["width"] <= b["x"] or b["x"] + b["width"] <= a["x"] or
                    a["y"] + a["height"] <= b["y"] or b["y"] + b["height"] <= a["y"])

    # Beam search chooses a globally better combination than greedy placement.
    # Each state is (score, selected pieces); overlap-free states survive pruning.
    beam = [(0.0, [])]
    for candidate in candidates:
        expanded = list(beam)
        for score, selected in beam:
            if len(selected) >= 20 or any(overlaps(candidate, piece) for piece in selected):
                continue
            expanded.append((score + candidate["value"], selected + [candidate]))
        expanded.sort(key=lambda state: state[0], reverse=True)
        beam = expanded[:180]
    pieces = beam[0][1] if beam else []

    # Recreate the reference treatment: solid coral paper with original
    # worksheet marks preserved as darker coral instead of washed-out pink.
    coral = np.array([91.0, 85.0, 242.0], dtype=np.float32)       # #F2555B BGR
    dark_coral = np.array([49.0, 41.0, 156.0], dtype=np.float32)
    darkness = 1.0 - (gray.astype(np.float32) / 255.0)
    darkness = np.clip(darkness * 0.72, 0.0, 0.72)[..., None]
    annotated = np.clip(coral * (1.0 - darkness) + dark_coral * darkness, 0, 255).astype(np.uint8)
    for piece in pieces:
        x, y, w, h = piece["x"], piece["y"], piece["width"], piece["height"]
        piece_color = products[piece["product"]][3]
        annotated[y:y + h, x:x + w] = np.array(piece_color, dtype=np.uint8)
        border = tuple(max(0, int(channel * .72)) for channel in piece_color)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), border, 5)
        title = "Mini Sticky" if piece["product"] == "mini_sticky" else piece["label"]
        dimensions = piece["size_in"]
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        scale = max(0.35, min(1.25, min(w / max(1, len(title) * 24), h / 90)))
        title_size = cv2.getTextSize(title, font, scale, thickness)[0]
        tx = x + max(5, (w - title_size[0]) // 2)
        ty = y + max(title_size[1] + 8, (h + title_size[1]) // 2 - 10)
        dim_scale = max(.4, scale * .65)
        dim_size = cv2.getTextSize(dimensions, font, dim_scale, 1)[0]
        cv2.putText(annotated, title, (tx, ty), font, scale, (10, 10, 10), thickness, cv2.LINE_AA)
        dx = x + max(5, (w - dim_size[0]) // 2)
        cv2.putText(annotated, dimensions, (dx, ty + dim_size[1] + 8), font, dim_scale, (10, 10, 10), 1, cv2.LINE_AA)
    # Keep the page boundary clearly visible in the final plan.
    cv2.rectangle(annotated, (3, 3), (width - 4, height - 4), (0, 0, 0), 6)
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

    # Normalize uneven lighting before thresholding. This reduces false ink caused
    # by camera shadows while preserving faint gray worksheet text.
    background = cv2.GaussianBlur(roi, (0, 0), 21)
    normalized = cv2.divide(roi, background, scale=255)
    adaptive_ink = cv2.adaptiveThreshold(normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 35, 10)
    # A phone photo can make blank paper gray. Only substantially dark pixels
    # count as printed content; brightness normalization handles shadows.
    _, raw_ink = cv2.threshold(roi, 155, 255, cv2.THRESH_BINARY_INV)
    threshold = cv2.bitwise_or(adaptive_ink, raw_ink)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
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
    _, plan_light_ink = cv2.threshold(normalized, 220, 255, cv2.THRESH_BINARY_INV)
    plan_light_ink = cv2.morphologyEx(plan_light_ink, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    plan_threshold = cv2.bitwise_or(threshold, plan_light_ink)
    full_ink[margin_y:height - margin_y, margin_x:width - margin_x] = plan_threshold
    # Usable area is practical cuttable paper, not simply white pixels. Remove
    # a clearance around all detected ink and the outer sheet border first.
    safe_ink = cv2.dilate((full_ink > 0).astype(np.uint8),
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)))
    safe_ink[[0, 1, -2, -1], :] = 1
    safe_ink[:, [0, 1, -2, -1]] = 1
    interior = safe_ink[margin_y:height - margin_y, margin_x:width - margin_x]
    usable = round(float(np.count_nonzero(interior == 0)) / max(1, interior.size) * 100, 1)
    plan_image, pieces, plan_status = _make_cutting_plan(sheet, full_ink, demand or {})
    sheet_ok, sheet_encoded = cv2.imencode(".jpg", sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    sheet_image = "data:image/jpeg;base64," + base64.b64encode(sheet_encoded.tobytes()).decode("ascii") if sheet_ok else ""
    return {
        "blank": usable,
        "unusable": round(100 - usable, 1),
        "printed": printed,
        "damage": damage,
        "confidence": confidence,
        "type": "document detected" if detection_confidence > 60 else "best-effort scan",
        "width": int(width),
        "height": int(height),
        "sheet_image": sheet_image,
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


@app.post("/api/cutting/gcode")
def cutting_gcode():
    """Return reviewed, machine-space G-code without sending it to a printer."""
    payload = request.get_json(silent=True) or {}
    try:
        config_data = payload.get("config") or {}
        config = CutterConfig(
            paper_width_mm=float(config_data.get("paper_width_mm", 210.0)),
            paper_height_mm=float(config_data.get("paper_height_mm", 297.0)),
            origin_x_mm=float(config_data.get("origin_x_mm", 0.0)),
            origin_y_mm=float(config_data.get("origin_y_mm", 0.0)),
            max_x_mm=float(config_data.get("max_x_mm", 290.0)),
            max_y_mm=float(config_data.get("max_y_mm", 300.0)),
            safe_z_mm=float(config_data.get("safe_z_mm", 5.0)),
            blade_z_mm=float(config_data.get("blade_z_mm", 0.0)),
            travel_feed=int(config_data.get("travel_feed", 2400)),
            cut_feed=int(config_data.get("cut_feed", 600)),
            passes=int(config_data.get("passes", 1)),
        )
        gcode = generate_gcode(payload.get("pieces", []), payload["image_width"], payload["image_height"], config)
        return jsonify(gcode=gcode, piece_count=len(payload.get("pieces", [])))
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify(error=str(exc)), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
