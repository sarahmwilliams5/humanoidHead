"""
Audio2Expression (ARKit 52) Arduino Servo Driver
================================================
This script converts an audio file into 52 ARKit blendshape expressions using
the Audio2Expression model (Wav2Vec2 + LAM Audio2Expression fused ONNX), then
uses a specified blendshape (default: 'jawOpen') to drive a servo motor on an
Arduino board via pyFirmata in real-time with synchronized audio playback.

Dependencies:
    pip install pyfirmata2 pyserial onnxruntime soundfile scipy numpy huggingface_hub
"""

import os
import sys
import time
import argparse
import numpy as np
import soundfile as sf
import scipy.signal as signal
import onnxruntime as ort

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Optional winsound for synchronized audio playback on Windows
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

# Arduino Firmata interface
try:
    import pyfirmata2
    import serial.tools.list_ports
    HAS_PYFIRMATA = True
except ImportError:
    HAS_PYFIRMATA = False

# ---------------------------------------------------------
# Default Configuration
# ---------------------------------------------------------
DEFAULT_AUDIO_PATH = "sample_speech.wav"     # Relative path to input audio file
DEFAULT_MODEL_PATH = "wav2arkit_cpu.onnx"     # ONNX Audio2Expression model file
DEFAULT_PORT = None                          # None = auto-detect, or 'COM3', '/dev/ttyACM0'
DEFAULT_PIN = "d:9:s"                        # Digital pin 9 in servo mode
DEFAULT_BLENDSHAPE = "jawOpen"               # Blendshape used to drive the servo
DEFAULT_MIN_ANGLE = 0                        # Servo angle (degrees) when blendshape is 0.0
DEFAULT_MAX_ANGLE = 180                       # Servo angle (degrees) when blendshape is 1.0 (or max)
DEFAULT_GAIN = 3.5                           # Multiplier to scale conversational speech to full movement
DEFAULT_SMOOTHING = 0.25                     # Exponential moving average factor (0.0=off, 0.9=heavy)
DEFAULT_FPS = 30.0                           # Output frame rate produced by Audio2Expression model

# Complete list of 52 ARKit blendshapes in standard order
ARKIT52_NAMES = [
    "browDownLeft",         # 0
    "browDownRight",        # 1
    "browInnerUp",          # 2
    "browOuterUpLeft",      # 3
    "browOuterUpRight",     # 4
    "cheekPuff",            # 5
    "cheekSquintLeft",      # 6
    "cheekSquintRight",     # 7
    "eyeBlinkLeft",         # 8
    "eyeBlinkRight",        # 9
    "eyeLookDownLeft",      # 10
    "eyeLookDownRight",     # 11
    "eyeLookInLeft",        # 12
    "eyeLookInRight",       # 13
    "eyeLookOutLeft",       # 14
    "eyeLookOutRight",      # 15
    "eyeLookUpLeft",        # 16
    "eyeLookUpRight",       # 17
    "eyeSquintLeft",        # 18
    "eyeSquintRight",       # 19
    "eyeWideLeft",          # 20
    "eyeWideRight",         # 21
    "jawForward",           # 22
    "jawLeft",              # 23
    "jawOpen",              # 24  <-- Default for mouth/jaw servo
    "jawRight",             # 25
    "mouthClose",           # 26
    "mouthDimpleLeft",      # 27
    "mouthDimpleRight",     # 28
    "mouthFrownLeft",       # 29
    "mouthFrownRight",      # 30
    "mouthFunnel",          # 31
    "mouthLeft",            # 32
    "mouthLowerDownLeft",   # 33
    "mouthLowerDownRight",  # 34
    "mouthPressLeft",       # 35
    "mouthPressRight",      # 36
    "mouthPucker",          # 37
    "mouthRight",           # 38
    "mouthRollLower",       # 39
    "mouthRollUpper",       # 40
    "mouthShrugLower",      # 41
    "mouthShrugUpper",      # 42
    "mouthSmileLeft",       # 43
    "mouthSmileRight",      # 44
    "mouthStretchLeft",     # 45
    "mouthStretchRight",    # 46
    "mouthUpperUpLeft",     # 47
    "mouthUpperUpRight",    # 48
    "noseSneerLeft",        # 49
    "noseSneerRight",       # 50
    "tongueOut"             # 51
]


def ensure_model_files(model_path: str):
    """Ensure the ONNX model and external weights file exist locally."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_full = os.path.join(script_dir, model_path) if not os.path.isabs(model_path) else model_path
    data_full = f"{model_full}.data"

    if not (os.path.exists(model_full) and os.path.exists(data_full)):
        print(f"[Model] Downloading Audio2Expression model (myned-ai/wav2arkit_cpu) to '{script_dir}'...")
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download(repo_id="myned-ai/wav2arkit_cpu", filename="wav2arkit_cpu.onnx", local_dir=script_dir)
            hf_hub_download(repo_id="myned-ai/wav2arkit_cpu", filename="wav2arkit_cpu.onnx.data", local_dir=script_dir)
            print("[Model] Download completed.")
        except Exception as exc:
            raise RuntimeError(f"Failed to auto-download model files: {exc}")

    return model_full


def load_and_preprocess_audio(audio_path: str, target_sr: int = 16000) -> np.ndarray:
    """Load relative audio file, convert to mono float32, and resample to 16 kHz."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio_data, sr = sf.read(audio_path, dtype="float32")

    # Convert multi-channel to mono if necessary
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    # Resample to 16,000 Hz if necessary
    if sr != target_sr:
        # Use polyphase filtering for fast and high-quality resampling
        gcd = np.gcd(int(target_sr), int(sr))
        up = int(target_sr // gcd)
        down = int(sr // gcd)
        audio_data = signal.resample_poly(audio_data, up, down).astype(np.float32)

    # Normalize audio levels
    peak = np.max(np.abs(audio_data))
    if peak > 0:
        audio_data = audio_data / peak

    return audio_data


def predict_arkit52(audio_waveform: np.ndarray, model_path: str) -> np.ndarray:
    """
    Run Audio2Expression ONNX inference.
    Input waveform: 1D array of 16kHz audio samples.
    Returns: numpy array of shape (frames, 52) with values between [0, 1].
    """
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inp_tensor = audio_waveform.reshape(1, -1).astype(np.float32)
    output = session.run(None, {"audio_waveform": inp_tensor})[0]
    # Output shape: [1, num_frames, 52]
    return output[0]


def auto_detect_arduino_port():
    """Attempt to detect a connected Arduino COM port on Windows/Linux."""
    if not HAS_PYFIRMATA:
        return None
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        desc = (p.description or "").lower()
        if "arduino" in desc or "ch340" in desc or "usb-serial" in desc or "ttyacm" in p.device.lower():
            return p.device
    if len(ports) == 1:
        return ports[0].device
    return None


def normalize_pin_str(pin_input: str) -> str:
    """Normalize user input like '9', 'D9', or 'd:9:s' into standard pyFirmata format."""
    pin_clean = str(pin_input).strip().lower()
    if pin_clean.isdigit():
        return f"d:{pin_clean}:s"
    if pin_clean.startswith("d") and pin_clean[1:].isdigit():
        return f"d:{pin_clean[1:]}:s"
    return pin_clean


def connect_arduino(port=None, pin_str=DEFAULT_PIN):
    """
    Connect to Arduino board via pyFirmata and configure the servo pin.
    Returns (board, servo) or (None, None) if unavailable.
    """
    if not HAS_PYFIRMATA:
        print("[Hardware] pyfirmata2/pyserial is not installed.")
        return None, None

    target_port = port or auto_detect_arduino_port()
    if not target_port:
        print("[Hardware] No serial/Arduino port detected.")
        return None, None

    norm_pin = normalize_pin_str(pin_str)
    try:
        print(f"[Hardware] Connecting to Arduino on port '{target_port}'...")
        board = pyfirmata2.Arduino(target_port)
        # Wait briefly for board initialization
        time.sleep(1.0)
        servo = board.get_pin(norm_pin)
        print(f"[Hardware] Connected successfully! Servo assigned to pin '{norm_pin}'.")
        return board, servo
    except Exception as exc:
        print(f"[Hardware] Could not connect to Arduino on '{target_port}': {exc}")
        return None, None


def render_ascii_bar(val: float, angle: float, width: int = 20) -> str:
    """Render a clean ASCII bar visualizer for terminal output."""
    filled = int(round(np.clip(val, 0.0, 1.0) * width))
    bar = "=" * filled + "-" * (width - filled)
    return f"[{bar}] {angle:3.0f} deg"


def play_audio_file(audio_path: str):
    """Asynchronously play the audio file on Windows using winsound."""
    if HAS_WINSOUND and audio_path.lower().endswith(".wav"):
        try:
            winsound.PlaySound(audio_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception:
            pass


def stop_audio():
    """Stop audio playback."""
    if HAS_WINSOUND:
        try:
            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass


def run_pipeline(
    audio_path=DEFAULT_AUDIO_PATH,
    model_path=DEFAULT_MODEL_PATH,
    port=DEFAULT_PORT,
    pin=DEFAULT_PIN,
    blendshape_name=DEFAULT_BLENDSHAPE,
    min_angle=DEFAULT_MIN_ANGLE,
    max_angle=DEFAULT_MAX_ANGLE,
    gain=DEFAULT_GAIN,
    smoothing=DEFAULT_SMOOTHING,
    fps=DEFAULT_FPS,
    simulate=False
):
    # Resolve relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(audio_path):
        audio_path = os.path.join(script_dir, audio_path)
    if not os.path.isabs(model_path):
        model_path = os.path.join(script_dir, model_path)

    # 1. Validate / download model
    model_path = ensure_model_files(model_path)

    # 2. Identify blendshape index
    if blendshape_name not in ARKIT52_NAMES:
        raise ValueError(
            f"Unknown ARKit blendshape: '{blendshape_name}'. "
            f"Choose from: {', '.join(ARKIT52_NAMES[:10])} ... and more (total 52)."
        )
    blendshape_idx = ARKIT52_NAMES.index(blendshape_name)

    # 3. Load & preprocess audio
    print(f"\n[Audio] Loading: {audio_path}")
    waveform = load_and_preprocess_audio(audio_path, target_sr=16000)
    audio_duration = len(waveform) / 16000.0
    print(f"[Audio] Resampled to 16 kHz mono. Duration: {audio_duration:.2f} seconds.")

    # 4. Predict ARKit 52 Blendshapes
    print(f"[Model] Generating ARKit 52 blendshapes using Audio2Expression...")
    t_start = time.perf_counter()
    blendshapes = predict_arkit52(waveform, model_path)
    t_inf = time.perf_counter() - t_start
    num_frames = blendshapes.shape[0]
    print(f"[Model] Done! Generated {num_frames} frames ({num_frames / audio_duration:.1f} fps) in {t_inf:.3f}s.")

    # Extract target blendshape track
    target_track = blendshapes[:, blendshape_idx]
    track_min = float(np.min(target_track))
    track_max = float(np.max(target_track))
    print(f"[ARKit] Selected blendshape #{blendshape_idx} ('{blendshape_name}'). Raw range: [{track_min:.3f}, {track_max:.3f}]")

    # 5. Connect to Arduino
    board = None
    servo = None
    if not simulate:
        board, servo = connect_arduino(port=port, pin_str=pin)

    if servo is None:
        if simulate:
            print("[Hardware] Running in SIMULATION mode (--simulate flag was specified).")
        else:
            print("[Hardware] WARNING: Could not connect to physical Arduino. Falling back to SIMULATION mode.")
    else:
        print(f"[Hardware] Driving servo on pin '{pin}' [Angles: {min_angle}° -> {max_angle}°]")

    # 6. Execute Synchronized Playback Loop
    frame_interval = 1.0 / fps
    smoothed_val = 0.0

    print(f"\n[Playback] Starting synchronized playback & servo animation (Press Ctrl+C to stop)...")
    print("-" * 70)

    # Start audio playback
    play_audio_file(audio_path)
    start_time = time.perf_counter()

    try:
        for frame_idx in range(num_frames):
            raw_val = float(target_track[frame_idx])

            # Apply gain and exponential moving average (EMA) smoothing
            scaled_val = np.clip(raw_val * gain, 0.0, 1.0)
            if frame_idx == 0:
                smoothed_val = scaled_val
            else:
                smoothed_val = (smoothing * smoothed_val) + ((1.0 - smoothing) * scaled_val)

            # Map blendshape [0.0, 1.0] to servo angle [min_angle, max_angle]
            angle = min_angle + smoothed_val * (max_angle - min_angle)
            angle = int(round(np.clip(angle, min(min_angle, max_angle), max(min_angle, max_angle))))

            # Send command to Arduino servo
            if servo is not None:
                servo.write(angle)

            # Display real-time progress
            elapsed = time.perf_counter() - start_time
            visual_bar = render_ascii_bar(smoothed_val, angle)
            sys.stdout.write(
                f"\rFrame {frame_idx + 1:4d}/{num_frames:4d} [{elapsed:5.2f}s] "
                f"{blendshape_name}: {smoothed_val:5.3f} -> {visual_bar}"
            )
            sys.stdout.flush()

            # High-precision drift-free frame timing
            target_time = start_time + (frame_idx + 1) * frame_interval
            time_to_wait = target_time - time.perf_counter()
            if time_to_wait > 0:
                time.sleep(time_to_wait)

    except KeyboardInterrupt:
        print("\n[Playback] Stopped by user.")
    finally:
        stop_audio()
        # Reset servo to rest position and clean up
        if servo is not None:
            try:
                servo.write(min_angle)
                time.sleep(0.1)
            except Exception:
                pass
        if board is not None:
            try:
                board.exit()
            except Exception:
                pass
        print("\n[Playback] Completed. Servo reset to rest position.\n")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Audio2Expression to ARKit 52 single-servo Arduino driver."
    )
    parser.add_argument(
        "audio",
        nargs="?",
        default=DEFAULT_AUDIO_PATH,
        help=f"Relative path to audio file (default: '{DEFAULT_AUDIO_PATH}')."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help=f"Path to wav2arkit_cpu.onnx model (default: '{DEFAULT_MODEL_PATH}')."
    )
    parser.add_argument(
        "--port",
        default=DEFAULT_PORT,
        help="Arduino serial port (e.g., 'COM3' or '/dev/ttyACM0'). Defaults to auto-detect."
    )
    parser.add_argument(
        "--pin",
        default=DEFAULT_PIN,
        help=f"Arduino servo pin definition (default: '{DEFAULT_PIN}')."
    )
    parser.add_argument(
        "--blendshape",
        default=DEFAULT_BLENDSHAPE,
        help=f"Target ARKit 52 blendshape (default: '{DEFAULT_BLENDSHAPE}')."
    )
    parser.add_argument(
        "--min-angle",
        type=int,
        default=DEFAULT_MIN_ANGLE,
        help=f"Minimum servo angle in degrees when closed (default: {DEFAULT_MIN_ANGLE})."
    )
    parser.add_argument(
        "--max-angle",
        type=int,
        default=DEFAULT_MAX_ANGLE,
        help=f"Maximum servo angle in degrees when open (default: {DEFAULT_MAX_ANGLE})."
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=DEFAULT_GAIN,
        help=f"Sensitivity multiplier (default: {DEFAULT_GAIN})."
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=DEFAULT_SMOOTHING,
        help=f"Smoothing factor between 0.0 (raw) and 0.9 (heavy) (default: {DEFAULT_SMOOTHING})."
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run without attempting to connect to physical Arduino hardware."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(
        audio_path=args.audio,
        model_path=args.model,
        port=args.port,
        pin=args.pin,
        blendshape_name=args.blendshape,
        min_angle=args.min_angle,
        max_angle=args.max_angle,
        gain=args.gain,
        smoothing=args.smoothing,
        simulate=args.simulate
    )
