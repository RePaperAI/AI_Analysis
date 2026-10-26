
import serial
import time

PORT = "COM8"
BAUD_RATE = 115200

try:
    with serial.Serial(PORT, BAUD_RATE, timeout=1) as printer:
        print("Connected. Waiting for printer...")
        time.sleep(3)  # Allow the printer to reset after connection

        printer.reset_input_buffer()

        print("Sending M115...")
        printer.write(b"M115\n")
        printer.flush()

        # Read the response line by line
        start = time.time()
        received = False

        while time.time() - start < 8:
            line = printer.readline().decode(
                "utf-8", errors="replace"
            ).strip()

            if line:
                print(line)
                received = True

                if line.lower() == "ok":
                    break

        if not received:
            print("No response received. Check the connection.")

except serial.SerialException as error:
    print("Serial connection error:", error)
