# RePaper AI scanner

Run the real OpenCV-backed dashboard with:

```powershell
python -m pip install -r requirements.txt
python scanner.py
```

Open http://127.0.0.1:5000. Upload a JPG, PNG, or WebP image. The service detects the paper boundary, corrects perspective, measures dark/printed pixels, estimates large damaged regions, and sends the percentages back to the dashboard.

Product sizes, colors, and default demand values are configured in `pieces_config.json`.

The cutting module generates relative perimeter G-code for a blade-equipped printer using A4 dimensions (210 x 297 mm portrait or 297 x 210 mm landscape). The printer's current position is the origin. Place the paper's bottom-right corner there; X increases toward the left and Y increases upward. The configured printer limits are `X=0..310 mm`, `Y=0..300 mm`. Automated cutting travels at pen-up `Z=44 mm` and lowers to pen-down `Z=41 mm` only during perimeter cuts. The machine must be homed and positioned at pen-up height before executing the relative file; test with the blade raised before cutting.

The configured maximum XY feed is 18,000 mm/min (300 mm/s), with Z at 300 mm/min. Cutting uses the configured `cut_feed`; maximum speed is not automatically safe for every blade or paper type, so test with the blade raised first.

The percentages are computer-vision estimates. For best results, photograph one sheet against a contrasting background with even lighting and all four edges visible.

To create a validated absolute move without connecting to the printer:

```python
from machine_control import coordinate_command

print(coordinate_command(100, 50, 44))
# G1 X100.00 Y50.00 Z44.00 F2400
```

Call `move_to_coordinate(100, 50, 44)` only after the printer is homed and the paper bottom-right is aligned with the machine origin; it sends the command and waits for the printer's `ok` response.
