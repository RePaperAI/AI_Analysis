"""Generate safe, machine-space G-code for a blade-equipped 3D printer.

The module only generates or explicitly sends G-code when called. It never
homes or moves a printer automatically from the dashboard.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CutterConfig:
    # A4 paper: 210 x 297 mm.
    paper_width_mm: float = 210.0
    paper_height_mm: float = 297.0
    # The lower-left corner of the A4 sheet is the machine origin.
    origin_x_mm: float = 0.0
    origin_y_mm: float = 0.0
    # Printer travel limits.
    max_x_mm: float = 290.0
    max_y_mm: float = 300.0
    safe_z_mm: float = 5.0
    blade_z_mm: float = 0.0
    travel_feed: int = 2400
    cut_feed: int = 600
    passes: int = 1


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

    Image coordinates start at the top-left. Machine coordinates use the
    paper's bottom-left as (0, 0): X increases rightward and Y is inverted. The
    generated file uses relative moves, so the printer's current position is
    treated as the origin when the file starts.
    """
    cfg = config or CutterConfig()
    image_width = _number(image_width, "image_width")
    image_height = _number(image_height, "image_height")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Scanned sheet dimensions must be positive")
    if cfg.passes < 1:
        raise ValueError("passes must be at least 1")
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

    rectangles = [_validate_piece(piece, image_width, image_height) for piece in pieces]
    for index, current in enumerate(rectangles):
        if any(_overlap(current, other) for other in rectangles[index + 1:]):
            raise ValueError("Cutting pieces overlap; refusing to generate unsafe G-code")

    def mx(px: float) -> float:
        return px / image_width * cfg.paper_width_mm

    def my(py: float) -> float:
        return cfg.paper_height_mm - py / image_height * cfg.paper_height_mm

    def move(axis: str, distance: float, feed: int) -> str:
        sign = "+" if distance >= 0 else ""
        return f"G0 {axis}{sign}{distance:.2f} F{feed}"

    def cut_move(axis: str, distance: float, feed: int) -> str:
        sign = "+" if distance >= 0 else ""
        return f"G1 {axis}{sign}{distance:.2f} F{feed}"

    lines = [
        "; RePaper AI blade cutting plan",
        "; Relative measurement mode: current printer position is X0 Y0 Z0",
        "; Place the paper bottom-left corner at the current X/Y position",
        "; Review paper placement, blade depth, and feed rates before running",
        "G21 ; millimetres",
        "G91 ; relative positioning",
        "M5 ; blade/spindle off if supported",
        move("Z", cfg.safe_z_mm, cfg.travel_feed),
    ]
    current_x = 0.0
    current_y = 0.0
    for index, (x, y, width, height) in enumerate(rectangles, start=1):
        left, right = mx(x), mx(x + width)
        top, bottom = my(y), my(y + height)
        path = [(left, top), (right, top), (right, bottom), (left, bottom), (left, top)]
        lines.extend([
            f"; Piece {index}",
            f"; Size {width / image_width * cfg.paper_width_mm:.2f} x {height / image_height * cfg.paper_height_mm:.2f} mm",
            move("Z", 0.0, cfg.travel_feed),
        ])
        first_x, first_y = path[0]
        lines.append(f"G0 X{first_x - current_x:+.2f} Y{first_y - current_y:+.2f} F{cfg.travel_feed}")
        current_x, current_y = first_x, first_y
        for pass_number in range(1, cfg.passes + 1):
            lines.extend([
                f"; pass {pass_number}",
                cut_move("Z", cfg.blade_z_mm - cfg.safe_z_mm, cfg.cut_feed),
            ])
            for target_x, target_y in path[1:]:
                lines.append(f"G1 X{target_x - current_x:+.2f} Y{target_y - current_y:+.2f} F{cfg.cut_feed}")
                current_x, current_y = target_x, target_y
        lines.append(cut_move("Z", cfg.safe_z_mm - cfg.blade_z_mm, cfg.travel_feed))
    lines.extend([
        cut_move("Z", -cfg.safe_z_mm, cfg.travel_feed),
        f"G0 X{-current_x:+.2f} Y{-current_y:+.2f} F{cfg.travel_feed}",
        "; End of plan",
        "M5",
    ])
    return "\n".join(lines) + "\n"
