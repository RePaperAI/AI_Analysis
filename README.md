# RePaper AI scanner

Run the real OpenCV-backed dashboard with:

```powershell
python -m pip install -r requirements.txt
python scanner.py
```

Open http://127.0.0.1:5000. Upload a JPG, PNG, or WebP image. The service detects the paper boundary, corrects perspective, measures dark/printed pixels, estimates large damaged regions, and sends the percentages back to the dashboard.

Product sizes, colors, and default demand values are configured in `pieces_config.json`.

The percentages are computer-vision estimates. For best results, photograph one sheet against a contrasting background with even lighting and all four edges visible.
