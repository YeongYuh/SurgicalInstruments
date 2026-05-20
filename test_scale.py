import serial
import time

PORTS = ["/dev/ttyUSB0", "/dev/ttyACM0"]
BAUDRATES = [9600, 115200]

for port in PORTS:
    for baudrate in BAUDRATES:
        try:
            print(f"\nTrying {port} @ {baudrate} ...")
            ser = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2)

            print(f"Connected to {port} @ {baudrate}")
            print("Reading 20 lines...")

            got_data = False
            for i in range(20):
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                print(f"{i + 1}: {line}")
                if line:
                    got_data = True

            ser.close()

            if got_data:
                print(f"\nSUCCESS: scale data found on {port} @ {baudrate}")
                raise SystemExit(0)

        except Exception as e:
            print(f"Failed {port} @ {baudrate}: {e}")

print("\nNo readable scale data found.")
