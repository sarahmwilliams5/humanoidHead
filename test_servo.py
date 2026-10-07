"""
Servo Quick-Test Utility
========================
Test and verify physical Arduino servo connection independently of audio inference.
"""

import sys
import time
import argparse
import pyfirmata2
import serial.tools.list_ports


def auto_detect_arduino_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        desc = (p.description or "").lower()
        if "arduino" in desc or "ch340" in desc or "usb-serial" in desc or "ttyacm" in p.device.lower():
            return p.device
    if len(ports) == 1:
        return ports[0].device
    return None


def test_servo(port=None, pin=9, min_angle=0, max_angle=90, cycles=3):
    target_port = port or auto_detect_arduino_port()
    if not target_port:
        print("[Error] No Arduino port detected. Please specify --port manually.")
        sys.exit(1)

    print(f"Connecting to Arduino on {target_port}...")
    board = pyfirmata2.Arduino(target_port)
    time.sleep(1.0)

    pin_str = f"d:{pin}:s"
    print(f"Attaching servo to Digital Pin {pin} ({pin_str})...")
    servo = board.get_pin(pin_str)

    try:
        print(f"\nStarting test: sweeping between {min_angle}° and {max_angle}° ({cycles} cycles)...")
        for i in range(1, cycles + 1):
            print(f"Cycle {i}/{cycles}: Moving to {max_angle}°...")
            servo.write(max_angle)
            time.sleep(1.0)

            print(f"Cycle {i}/{cycles}: Moving to {min_angle}°...")
            servo.write(min_angle)
            time.sleep(1.0)

        print("\n[Success] Servo test completed!")
    except KeyboardInterrupt:
        print("\nTest cancelled by user.")
    finally:
        try:
            servo.write(min_angle)
            time.sleep(0.2)
        except Exception:
            pass
        board.exit()
        print("Disconnected from Arduino.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Arduino servo connection.")
    parser.add_argument("--port", default=None, help="COM port (e.g., COM4)")
    parser.add_argument("--pin", type=int, default=9, help="Digital pin number (default: 9)")
    parser.add_argument("--min-angle", type=int, default=0, help="Min angle (default: 0)")
    parser.add_argument("--max-angle", type=int, default=90, help="Max angle (default: 90)")
    parser.add_argument("--cycles", type=int, default=3, help="Number of sweep cycles (default: 3)")
    args = parser.parse_args()

    test_servo(port=args.port, pin=args.pin, min_angle=args.min_angle, max_angle=args.max_angle, cycles=args.cycles)
