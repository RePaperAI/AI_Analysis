"""Generate safe, machine-space G-code for a blade-equipped 3D printer.

The module only generates or explicitly sends G-code when called. It never
homes or moves a printer automatically from the dashboard.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class CutterConfig:
    # Current sheet: landscape A4, 297 x 210 mm.
    paper_width_mm: float = 297.0
    paper_height_mm: float = 210.0
    # Paper corner coordinates in the machine's home-based coordinate frame.
    origin_x_mm: float = 0.0
    origin_y_mm: float = 0.0
    top_right_x_mm: float = 5.0
    top_right_y_mm: float = 53.0
    bottom_right_x_mm: float = 5.0
    bottom_right_y_mm: float = 263.0
    bottom_left_x_mm: float = 302.0
    bottom_left_y_mm: float = 263.0
    # Printer travel limits.
    max_x_mm: float = 310.0
    max_y_mm: float = 300.0
    safe_z_mm: float = 5.0
    blade_z_mm: float = 0.0
    pen_up_z_mm: float = 44.0
    pen_down_z_mm: float = 41.0
    travel_feed: int = 18000
    cut_feed: int = 18000
    z_feed: int = 300
    passes: int = 1


COORDINATES_CONFIG_PATH = Path(__file__).with_name("coordinates_config.json")


def load_cutter_config(path: str | Path | None = None) -> CutterConfig:
    """Load the machine coordinate settings from coordinates_config.json."""
    config_path = Path(path) if path else COORDINATES_CONFIG_PATH
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        paper = data["paper"]
        printer = data["printer"]
        tool = data["tool"]
        if paper.get("origin_corner") != "bottom_right":
            raise ValueError("coordinates_config.json must use bottom_right as origin_corner")
        if paper.get("orientation") != "landscape":
            raise ValueError("coordinates_config.json must use landscape paper orientation")
        return CutterConfig(
            paper_width_mm=float(paper["width_mm"]),
            paper_height_mm=float(paper["height_mm"]),
            origin_x_mm=float(paper.get("origin_x_mm", 0.0)),
            origin_y_mm=float(paper.get("origin_y_mm", 0.0)),
            top_right_x_mm=float(paper["machine_corners_mm"]["top_right"]["x"]),
            top_right_y_mm=float(paper["machine_corners_mm"]["top_right"]["y"]),
            bottom_right_x_mm=float(paper["machine_corners_mm"]["bottom_right"]["x"]),
            bottom_right_y_mm=float(paper["machine_corners_mm"]["bottom_right"]["y"]),
            bottom_left_x_mm=float(paper["machine_corners_mm"]["bottom_left"]["x"]),
            bottom_left_y_mm=float(paper["machine_corners_mm"]["bottom_left"]["y"]),
            max_x_mm=float(printer["max_x_mm"]),
            max_y_mm=float(printer["max_y_mm"]),
            pen_up_z_mm=float(tool["pen_up_z_mm"]),
            pen_down_z_mm=float(tool["pen_down_z_mm"]),
            travel_feed=int(tool.get("travel_feed", 2400)),
            cut_feed=int(tool.get("cut_feed", 18000)),
            z_feed=int(tool.get("z_feed", 300)),
            passes=int(tool.get("passes", 1)),
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid coordinate configuration: {config_path}: {exc}") from exc


def _number(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if result != result or result in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_piece(piece: dict, image_width: float, image_height: float) -> tuple[float, float, float, float]:
    x = _number(piece.get("x"), "piece.x")
    y = _number(piece.get("y"), "piece.y")
    width = _number(piece.get("width"), "piece.width")
    height = _number(piece.get("height"), "piece.height")
    if width <= 0 or height <= 0 or x < 0 or y < 0 or x + width > image_width or y + height > image_height:
        raise ValueError("A cutting piece is outside the scanned sheet")
    return x, y, width, height


def _overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


def generate_gcode(pieces: Iterable[dict], image_width: float, image_height: float,
                   config: CutterConfig | None = None) -> str:
    """Convert image-space rectangles into relative perimeter-cut G-code.

    Image coordinates start at the top-left. The configured machine frame maps
    the current landscape sheet as follows:

        paper top-right    = (5, 53)
        paper top-left     = (302, 53)
        paper bottom-left  = (302, 263)
        paper bottom-right = (5, 263)

    X increases toward the paper's left and Y increases toward the paper's
    bottom. The generated file uses relative moves after homing, with home
    position (0,0) as the machine base.
    """
    cfg = config or load_cutter_config()
    image_width = _number(image_width, "image_width")
    image_height = _number(image_height, "image_height")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Scanned sheet dimensions must be positive")
    if cfg.passes < 1:
        raise ValueError("passes must be at least 1")
    if cfg.pen_down_z_mm >= cfg.pen_up_z_mm:
        raise ValueError("Pen-down Z must be lower than pen-up Z")
    if cfg.paper_width_mm <= 0 or cfg.paper_height_mm <= 0:
        raise ValueError("Paper dimensions must be positive")
    if cfg.origin_x_mm < 0 or cfg.origin_y_mm < 0:
        raise ValueError("The paper origin cannot be negative")
    if cfg.max_x_mm <= 0 or cfg.max_y_mm <= 0:
        raise ValueError("Printer travel limits must be positive")
    if cfg.origin_x_mm + cfg.paper_width_mm > cfg.max_x_mm:
        raise ValueError("A4 paper exceeds the printer X travel limit")
    if cfg.origin_y_mm + cfg.paper_height_mm > cfg.max_y_mm:
        raise ValueError("A4 paper exceeds the printer Y travel limit")

    source_pieces = list(pieces)
    rectangles = [_validate_piece(piece, image_width, image_height) for piece in source_pieces]
    for index, current in enumerate(rectangles):
        if any(_overlap(current, other) for other in rectangles[index + 1:]):
            raise ValueError("Cutting pieces overlap; refusing to generate unsafe G-code")

    def mx(px: float) -> float:
        # Camera image left is machine X toward the paper's left.
        return cfg.bottom_left_x_mm - px / image_width * cfg.paper_width_mm

    def my(py: float) -> float:
        # Camera image top is machine Y=53; image bottom is machine Y=263.
        return cfg.top_right_y_mm + py / image_height * cfg.paper_height_mm

    def move(axis: str, distance: float, feed: int) -> str:
        sign = "+" if distance >= 0 else ""
        return f"G0 {axis}{sign}{distance:.2f} F{feed}"

    def cut_move(axis: str, distance: float, feed: int) -> str:
        sign = "+" if distance >= 0 else ""
        return f"G1 {axis}{sign}{distance:.2f} F{feed}"

    lines = [
        "; RePaper AI blade cutting plan",
        "; Startup: home X/Y, home Z, raise pen; paper bottom-right is the origin",
        "; Review paper placement, blade depth, and feed rates before running",
        "G21 ; millimetres",
        "G90 ; absolute setup positioning",
        "G28 X Y ; home X and Y",
        "G28 Z ; home Z after X/Y",
        f"G1 Z{cfg.pen_up_z_mm:.2f} F{cfg.z_feed} ; pen up",
        "G91 ; relative cutting measurements from paper bottom-right (0,0)",
        "M5 ; blade/spindle off if supported",
        f"; Pen up height Z{cfg.pen_up_z_mm:.2f} mm; pen down height Z{cfg.pen_down_z_mm:.2f} mm",
        "; Homing must place the paper bottom-right at the machine position (0,0)",
    ]
    current_x = 0.0
    current_y = 0.0
    for index, ((x, y, width, height), source_piece) in enumerate(zip(rectangles, source_pieces), start=1):
        physical_width = _number(source_piece.get("width_mm", width / image_width * cfg.paper_width_mm), "piece.width_mm")
        physical_height = _number(source_piece.get("height_mm", height / image_height * cfg.paper_height_mm), "piece.height_mm")
        if physical_width <= 0 or physical_height <= 0:
            raise ValueError("Piece measurements must be positive")
        left = mx(x)
        top = my(y)
        # The measured printer frame has X increasing leftward and Y increasing
        # downward from the top edge of the paper.
        right = left - physical_width
        bottom = top + physical_height
        if right < cfg.origin_x_mm or bottom < cfg.origin_y_mm:
            raise ValueError("A cutting piece measurement exceeds the paper origin")
        path = [(left, top), (right, top), (right, bottom), (left, bottom), (left, top)]
        lines.extend([
            f"; Piece {index}",
            f"; Size {physical_width:.2f} x {physical_height:.2f} mm",
            move("Z", 0.0, cfg.travel_feed),
        ])
        first_x, first_y = path[0]
        lines.append(f"G0 X{first_x - current_x:+.2f} Y{first_y - current_y:+.2f} F{cfg.travel_feed}")
        current_x, current_y = first_x, first_y
        for pass_number in range(1, cfg.passes + 1):
            lines.extend([
                f"; pass {pass_number}",
                cut_move("Z", cfg.pen_down_z_mm - cfg.pen_up_z_mm, cfg.z_feed),
            ])
            for target_x, target_y in path[1:]:
                lines.append(f"G1 X{target_x - current_x:+.2f} Y{target_y - current_y:+.2f} F{cfg.cut_feed}")
                current_x, current_y = target_x, target_y
        lines.append(cut_move("Z", cfg.pen_up_z_mm - cfg.pen_down_z_mm, cfg.z_feed))
    lines.extend([
        # Stay at the raised/safe Z height while returning to the start.
        f"G0 X{-current_x:+.2f} Y{-current_y:+.2f} F{cfg.travel_feed}",
        "; End of plan",
        "M5",
    ])
    return "\n".join(lines) + "\n"
