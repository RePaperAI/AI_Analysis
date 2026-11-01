"""Small, guarded serial sender for the blade-equipped printer."""
from __future__ import annotations

import time
import logging
from cutting import CutterConfig, load_cutter_config

logger = logging.getLogger("repaper.machine")


PRINTER_PORT = "COM8"
PRINTER_BAUD = 115200


def home_printer(port: str = PRINTER_PORT, baud: int = PRINTER_BAUD,
                 config: CutterConfig | None = None) -> tuple[int, list[str]]:
    """Home X/Y, then Z, and raise the tool to the configured pen-up height."""
    cfg = config or load_cutter_config()
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError("Printer homing requires pyserial. Run: python -m pip install pyserial") from exc

    logs = [f"Connecting to {port} at {baud} baud", "Homing sequence: X/Y, then Z"]
    with serial.Serial(port, baud, timeout=1) as printer:
        logs.append("Connected; waiting for printer USB reset")
        time.sleep(3)
        printer.reset_input_buffer()
        _send_and_wait(printer, "G90", logs)
        _send_and_wait(printer, "G28 X Y", logs)
        _send_and_wait(printer, "G28 Z", logs)
        _send_and_wait(printer, f"G1 Z{cfg.pen_up_z_mm:.2f} F{cfg.z_feed}", logs)
    logs.append("Homing completed; tool is up")
    for entry in logs:
        print(entry, flush=True)
    return 4, logs


def prepare_camera_position(port: str = PRINTER_PORT, baud: int = PRINTER_BAUD,
                            config: CutterConfig | None = None) -> tuple[int, list[str]]:
    """Home the printer, raise Z to 50 mm, then move Y to 280 mm for scanning."""
    cfg = config or load_cutter_config()
    if 50.0 <= 0 or 280.0 > cfg.max_y_mm:
        raise ValueError("The configured camera position is outside printer travel")
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError("Camera-position automation requires pyserial. Run: python -m pip install pyserial") from exc

    logs = [f"Connecting to {port} at {baud} baud", "Automation: home X/Y, home Z, move Z50, then Y280"]
    with serial.Serial(port, baud, timeout=1) as printer:
        logs.append("Connected; waiting for printer USB reset")
        time.sleep(3)
        printer.reset_input_buffer()
        _send_and_wait(printer, "G90", logs)
        _send_and_wait(printer, "G28 X Y", logs)
        _send_and_wait(printer, "G28 Z", logs)
        _send_and_wait(printer, f"G1 Z50.00 F{cfg.z_feed}", logs)
        _send_and_wait(printer, f"G1 Y280.00 F{cfg.travel_feed}", logs)
        _send_and_wait(printer, "M400", logs)
    logs.append("Camera position ready: Z50, Y280")
    for entry in logs:
        print(entry, flush=True)
    return 5, logs


def coordinate_command(x_mm: float, y_mm: float, z_mm: float | None = None,
                      config: CutterConfig | None = None) -> str:
    """Build a safe absolute move for paper coordinates.

    Coordinates are absolute machine coordinates measured from the homed
    position. The configured paper corners are X5..302 and Y53..263.
    This function only creates a command; it does not connect to the printer.
    """
    cfg = config or load_cutter_config()
    try:
        x = float(x_mm)
        y = float(y_mm)
        z = None if z_mm is None else float(z_mm)
    except (TypeError, ValueError) as exc:
        raise ValueError("X, Y, and Z coordinates must be numeric") from exc
    if not (0.0 <= x <= cfg.max_x_mm):
        raise ValueError(f"X must be between 0 and {cfg.max_x_mm:g} mm")
    if not (0.0 <= y <= cfg.max_y_mm):
        raise ValueError(f"Y must be between 0 and {cfg.max_y_mm:g} mm")
    if z is not None and z < 0.0:
        raise ValueError("Z must not be negative")
    command = f"G1 X{x:.2f} Y{y:.2f}"
    if z is not None:
        command += f" Z{z:.2f}"
    return command + f" F{cfg.travel_feed}"


def move_to_coordinate(x_mm: float, y_mm: float, z_mm: float | None = None,
                       port: str = PRINTER_PORT, baud: int = PRINTER_BAUD,
                       config: CutterConfig | None = None) -> tuple[int, list[str]]:
    """Send one validated absolute paper-coordinate move to the printer.

    The printer must already be homed, with its home position aligned to the
    paper bottom-right corner. The move is sent with G90 and waits for ``ok``.
    """
    command = coordinate_command(x_mm, y_mm, z_mm, config)
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError("Printer movement requires pyserial. Run: python -m pip install pyserial") from exc
    logs = [f"Connecting to {port} at {baud} baud", f"Coordinate request: X{x_mm} Y{y_mm}" + (f" Z{z_mm}" if z_mm is not None else "")]
    with serial.Serial(port, baud, timeout=1) as printer:
        logs.append("Connected; waiting for printer USB reset")
        time.sleep(3)
        # Capture startup text for diagnostics, then discard stale responses
        # so G90 is matched to the acknowledgement for this request.
        startup_deadline = time.monotonic() + 1.0
        while time.monotonic() < startup_deadline:
            startup = printer.readline().decode("utf-8", errors="replace").strip()
            if startup:
                logs.append(f"startup: {startup}")
        printer.reset_input_buffer()
        logs.append("Sending coordinate move")
        _send_and_wait(printer, "G90", logs)
        _send_and_wait(printer, command, logs)
    logs.append("Coordinate move completed")
    for entry in logs:
        print(entry, flush=True)
    return 2, logs


def send_coordinates(coordinates: tuple[float, float, float],
                     port: str = PRINTER_PORT, baud: int = PRINTER_BAUD,
                     config: CutterConfig | None = None) -> tuple[int, list[str]]:
    """Home first, then send an ``(X, Y, Z)`` tuple from the home position.

    Example: ``send_coordinates((20, 30, 44))``.
    Coordinates are absolute machine coordinates from the homed position.
    Homing first makes the result independent of the printer's prior position.
    """
    if not isinstance(coordinates, (tuple, list)) or len(coordinates) != 3:
        raise ValueError("coordinates must contain exactly (X, Y, Z)")
    home_count, home_logs = home_printer(port, baud, config)
    move_count, move_logs = move_to_coordinate(coordinates[0], coordinates[1], coordinates[2], port, baud, config)
    return home_count + move_count, home_logs + move_logs


def _send_and_wait(printer, command: str, logs: list[str], timeout: float = 20.0) -> None:
    logs.append(f"> {command}")
    logger.info("> %s", command)
    printer.write((command.strip() + "\n").encode("ascii"))
    printer.flush()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = printer.readline().decode("utf-8", errors="replace").strip()
        if response.lower().startswith("error"):
            logs.append(f"ERROR: {response}")
            raise RuntimeError(f"Printer rejected {command}: {response}")
        if response:
            logs.append(response)
            logger.info("< %s", response)
        if response.lower() == "ok":
            return
    raise TimeoutError(f"Printer did not acknowledge: {command}")


def execute_gcode(program: str, port: str = PRINTER_PORT, baud: int = PRINTER_BAUD) -> tuple[int, list[str]]:
    """Send a generated program line-by-line and wait for every acknowledgement."""
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError("Printer execution requires pyserial. Run: python -m pip install pyserial") from exc
    commands = [line.strip() for line in program.splitlines()
                if line.strip() and not line.lstrip().startswith(";")]
    if not commands:
        raise ValueError("The G-code program is empty")
    logs = [f"Connecting to {port} at {baud} baud"]
    with serial.Serial(port, baud, timeout=1) as printer:
        logs.append("Connected; waiting for printer startup")
        time.sleep(3)
        printer.reset_input_buffer()
        for command in commands:
            _send_and_wait(printer, command, logs)
    logs.append(f"Completed {len(commands)} commands")
    return len(commands), logs
