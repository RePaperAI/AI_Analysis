"""Generate safe, machine-space G-code for a blade-equipped 3D printer.

The module only generates or explicitly sends G-code when called. It never
homes or moves a printer automatically from the dashboard.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CutterConfig:
    paper_width_mm: float = 216.0
    paper_height_mm: float = 279.0
    origin_x_mm: float = 10.0
    origin_y_mm: float = 10.0
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
    """Convert image-space rectangles into perimeter-cut G-code.

    Image coordinates start at the top-left. Machine coordinates start at the
    paper's lower-left, so Y is intentionally inverted during conversion.
    """
    cfg = config or CutterConfig()
    image_width = _number(image_width, "image_width")
    image_height = _number(image_height, "image_height")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Scanned sheet dimensions must be positive")
    if cfg.passes < 1:
        raise ValueError("passes must be at least 1")

    rectangles = [_validate_piece(piece, image_width, image_height) for piece in pieces]
    for index, current in enumerate(rectangles):
        if any(_overlap(current, other) for other in rectangles[index + 1:]):
            raise ValueError("Cutting pieces overlap; refusing to generate unsafe G-code")

    def mx(px: float) -> float:
        return cfg.origin_x_mm + px / image_width * cfg.paper_width_mm

    def my(py: float) -> float:
        return cfg.origin_y_mm + cfg.paper_height_mm - py / image_height * cfg.paper_height_mm

    lines = [
        "; RePaper AI blade cutting plan",
        "; Review bed size, paper placement, blade depth, and feed rates before running",
        "G21 ; millimetres",
        "G90 ; absolute positioning",
        "M5 ; blade/spindle off if supported",
        f"G0 Z{cfg.safe_z_mm:.2f} F{cfg.travel_feed}",
    ]
    for index, (x, y, width, height) in enumerate(rectangles, start=1):
        left, right = mx(x), mx(x + width)
        top, bottom = my(y), my(y + height)
        lines.extend([
            f"; Piece {index}",
            f"G0 Z{cfg.safe_z_mm:.2f} F{cfg.travel_feed}",
            f"G0 X{left:.2f} Y{top:.2f} F{cfg.travel_feed}",
        ])
        for pass_number in range(1, cfg.passes + 1):
            lines.extend([
                f"; pass {pass_number}",
                f"G1 Z{cfg.blade_z_mm:.2f} F{cfg.cut_feed}",
                f"G1 X{right:.2f} Y{top:.2f} F{cfg.cut_feed}",
                f"G1 X{right:.2f} Y{bottom:.2f} F{cfg.cut_feed}",
                f"G1 X{left:.2f} Y{bottom:.2f} F{cfg.cut_feed}",
                f"G1 X{left:.2f} Y{top:.2f} F{cfg.cut_feed}",
            ])
        lines.append(f"G0 Z{cfg.safe_z_mm:.2f} F{cfg.travel_feed}")
    lines.extend(["G0 X0 Y0 F2400", "; End of plan", "M5"])
    return "\n".join(lines) + "\n"
