# RePaper AI scanner

Run the real OpenCV-backed dashboard with:

```powershell
python -m pip install -r requirements.txt
python scanner.py
```

Open http://127.0.0.1:5000. Upload a JPG, PNG, or WebP image. The service detects the paper boundary, corrects perspective, measures dark/printed pixels, estimates large damaged regions, and sends the percentages back to the dashboard.

Product sizes, colors, and default demand values are configured in `pieces_config.json`.

The cutting module generates perimeter G-code for a blade-equipped printer. It assumes image Y=0 is the top of the paper and converts it to a lower-left machine origin. Review `CutterConfig` in `cutting.py` and test with the blade raised before cutting; the dashboard only generates/downloads G-code and does not move the printer automatically.

The percentages are computer-vision estimates. For best results, photograph one sheet against a contrasting background with even lighting and all four edges visible.
