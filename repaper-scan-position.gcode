; RePaper AI camera-position setup
; This file is not executed automatically by the scanner.
; Camera capture is performed by the dashboard, not by G-code.
; Camera capture is handled only by the dashboard.

G21 ; millimetres
G90 ; absolute positioning
; Printer must already be homed before this file is used.
G1 Z50.00 F300 ; raise tool for camera clearance
G1 Y280.00 F18000 ; move to camera position at maximum XY speed
M400 ; wait until all printer movement is complete
M5 ; tool off if supported

; Ready for camera capture
