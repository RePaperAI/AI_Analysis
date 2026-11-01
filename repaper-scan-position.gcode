; RePaper AI camera-position setup
; This file prepares the printer for the USB-camera scan.
; Camera capture is performed by the dashboard, not by G-code.
; Coordinates are absolute machine coordinates from the homed position.

G21 ; millimetres
G90 ; absolute positioning
G28 X Y ; home X and Y first
G28 Z ; home Z after X and Y
G1 Z50.00 F2400 ; raise tool for camera clearance
G1 Y280.00 F2400 ; move to camera position
M400 ; wait until all printer movement is complete
M5 ; tool off if supported

; Ready for camera capture
