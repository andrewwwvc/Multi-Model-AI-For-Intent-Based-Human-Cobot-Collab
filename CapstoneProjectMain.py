"""
Capstone Project - UR Robot Voice Command Interface
Step 1: Press SPACE to record a voice command
Step 2: spaCy matches the command semantically to a register value
Step 3: Write the register value to the UR robot via RTDE
        and poll until the robot finishes or reports an error
"""

import threading
import time
import speech_recognition as sr
import keyboard
import spacy
import rtde_receive
import rtde_io

# -- Configuration -------------------------------------------------------------
ROBOT_IP             = "192.168.1.112"   # <-- Your robot's IP
ENERGY_THRESHOLD     = 300               # Mic sensitivity
SIMILARITY_THRESHOLD = 0.70              # Minimum NLP match score (0.0 - 1.0)
POLL_RATE            = 0.5               # Seconds between register polls

# -- Register indices (must match your URScript) -------------------------------
REG_COMMAND        = 18
REG_STATUS         = 19
ERROR_PART_MISSING = 99

# -- Semantic Anchors ----------------------------------------------------------
ANCHORS = {
    1: [
        "emergency", "red", "stop",
        "place the red part",
        "place the red light",
        "assemble the red light",
        "install the red light",
        "pick up the red part",
        "put in the red light",
        "red light only",
    ],
    2: [
        "warning", "yellow", "caution",
        "place the yellow part",
        "place the yellow light",
        "assemble the yellow light",
        "install the yellow light",
        "pick up the yellow part",
        "put in the yellow light",
        "yellow light only",
    ],
    3: [
        "start", "green", "power",
        "place the green part",
        "place the green light",
        "assemble the green light",
        "install the green light",
        "pick up the green part",
        "put in the green light",
        "green light only",
    ],
    4: [
        "full", "all", "everything", "complete", "total", "entire",
        "full sequence",
        "entire assembly",
        "assemble everything",
        "place all lights",
        "place all parts",
        "run the full sequence",
        "do the full assembly",
        "assemble all three",
        "install all lights",
        "complete assembly",
        "run everything",
    ],
    0: [
        "cancel",
        "reset",
        "stop everything",
        "cancel all",
        "cancel all commands",
        "cancel all previous commands",
        "reset everything",
        "abort",
        "abort mission",
        "start over",
    ],
}

# -- NLP Setup -----------------------------------------------------------------
print("Loading spaCy model ...")
nlp = spacy.load("en_core_web_md")

ANCHOR_DOCS = {
    phrase: (reg_val, nlp(phrase))
    for reg_val, phrases in ANCHORS.items()
    for phrase in phrases
}

print("spaCy model loaded.\n")

# -- Speech Recognition Setup --------------------------------------------------
recogniser = sr.Recognizer()
recogniser.energy_threshold = ENERGY_THRESHOLD
recogniser.dynamic_energy_threshold = False

# -- Step 1: Speech-to-Text ----------------------------------------------------
def listen_for_command():
    """
    Wait for SPACE to start recording, SPACE again to stop.
    Returns the recognised text, or None if nothing was understood.
    """
    print("Press SPACE to start recording ...")
    keyboard.wait("space")
    print("Recording ... press SPACE to stop.")

    stop_event = threading.Event()
    keyboard.on_press_key("space", lambda _: stop_event.set())

    frames = []

    with sr.Microphone() as source:
        recogniser.adjust_for_ambient_noise(source, duration=0.5)

        while not stop_event.is_set():
            try:
                chunk = recogniser.record(source, duration=0.5)
                frames.append(chunk.get_raw_data())
            except Exception:
                break

        sample_rate = source.SAMPLE_RATE
        sample_width = source.SAMPLE_WIDTH

    keyboard.unhook_all()

    if not frames:
        print("No audio captured.")
        return None

    raw_audio = b"".join(frames)
    audio = sr.AudioData(raw_audio, sample_rate, sample_width)

    print("Processing speech-to-text ...")

    try:
        text = recogniser.recognize_google(audio)
        print(f"Heard: '{text}'")
        return text
    except sr.UnknownValueError:
        print("Could not understand audio - try again.")
        return None
    except sr.RequestError as e:
        print(f"STT request failed: {e}")
        return None

# -- Step 2: Semantic NLP Matching ---------------------------------------------
def get_best_match(user_input):
    """
    Compare the full voice command against every anchor phrase.
    Returns (register_value, matched_anchor, score).
    register_value is 0 if no anchor exceeds the threshold.
    """
    user_input_clean = user_input.lower()
    doc = nlp(user_input_clean)

    best_score = 0.0
    best_anchor = ""
    best_reg = 0

    for phrase, (reg_val, anchor_doc) in ANCHOR_DOCS.items():
        score = doc.similarity(anchor_doc)
        if score > best_score and score >= SIMILARITY_THRESHOLD:
            best_score = score
            best_anchor = phrase
            best_reg = reg_val

    return best_reg, best_anchor, best_score

# -- Step 3: RTDE Register Read/Write ------------------------------------------
def read_reg(rtde_r, index):
    return int(rtde_r.getOutputIntRegister(index))

def write_reg(rtde_io_, index, value):
    rtde_io_.setInputIntRegister(index, value)
    print(f"Register[{index}] -> {value}")

def wait_for_robot(rtde_r, rtde_io_):
    """
    Poll REG_STATUS (register 19) which the robot writes to signal completion:
      1  = robot is done successfully
      99 = robot encountered an error (part not found)
    Python resets both registers after receiving the signal.
    """
    print("Waiting for robot to finish ...")

    while True:
        status = read_reg(rtde_r, REG_STATUS)

        if status == 1:
            print("Robot finished successfully.")
            write_reg(rtde_io_, REG_COMMAND, 0)
            write_reg(rtde_io_, REG_STATUS, 0)
            return True

        elif status == ERROR_PART_MISSING:
            print("ERROR: Robot could not find a part.")
            write_reg(rtde_io_, REG_COMMAND, 0)
            write_reg(rtde_io_, REG_STATUS, 0)
            return False

        else:
            print("Robot busy ...")

        time.sleep(POLL_RATE)

def send_command(rtde_r, rtde_io_, reg_value):
    """
    Write command to register and wait for robot to finish.
    Returns True on success, False on error.
    """
    current = read_reg(rtde_r, REG_COMMAND)

    if current != 0:
        print(f"Robot is busy (register = {current}). Command not sent.")
        return False

    write_reg(rtde_io_, REG_COMMAND, reg_value)
    return wait_for_robot(rtde_r, rtde_io_)

# -- Main Loop -----------------------------------------------------------------
if __name__ == "__main__":
    print("=== Capstone Project - Voice Command Interface ===")
    print(f"Connecting to robot at {ROBOT_IP} ...")

    rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    rtde_io_ = rtde_io.RTDEIOInterface(ROBOT_IP)

    print("Connected to robot.\n")

    # Clear register at startup to avoid stale values from previous runs
    write_reg(rtde_io_, REG_COMMAND, 0)
    write_reg(rtde_io_, REG_STATUS, 0)
    print("Registers cleared.\n")

    try:
        print("Press SPACE to record a command. (Ctrl+C to quit)\n")
        print("Example commands:")
        print("  'place the red light'     -> register 1")
        print("  'place the yellow light'  -> register 2")
        print("  'place the green light'   -> register 3")
        print("  'run the full sequence'   -> register 4\n")

        while True:
            # Step 1 - Listen / STT
            user_text = listen_for_command()

            if user_text is None:
                print("Nothing heard - try again.\n")
                continue

            if user_text.lower() in ("quit", "exit", "stop listening"):
                print("Exit command heard. Goodbye!")
                break

            # Start timer AFTER recording/STT is done, BEFORE NLP matching begins
            nlp_start_time = time.perf_counter()

            # Step 2 - Match
            reg_value, matched_anchor, score = get_best_match(user_text)

            # Stop timer as soon as a key match to anchors is determined
            nlp_elapsed_time = time.perf_counter() - nlp_start_time

            if reg_value == 0 and matched_anchor == "cancel":
                print(f"Cancel command heard - resetting all registers.")
                print(f"NLP match time: {nlp_elapsed_time:.4f} seconds\n")
                write_reg(rtde_io_, REG_COMMAND, 0)
                write_reg(rtde_io_, REG_STATUS, 0)
                continue

            if reg_value == 0:
                print(
                    f"No confident match for '{user_text}' "
                    f"(below {SIMILARITY_THRESHOLD:.0%} threshold)."
                )
                print(f"NLP match time: {nlp_elapsed_time:.4f} seconds\n")
                continue

            print(
                f"Matched '{user_text}' -> anchor='{matched_anchor}' "
                f"(score={score:.2f}) -> register {reg_value}"
            )
            print(f"NLP match time: {nlp_elapsed_time:.4f} seconds")

            # Step 3 - Send to robot
            success = send_command(rtde_r, rtde_io_, reg_value)

            if success:
                print("Work order complete. Ready for next command.\n")
            else:
                print("Resetting register. Please check the robot.\n")
                write_reg(rtde_io_, REG_COMMAND, 0)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        rtde_r.disconnect()
        rtde_io_.disconnect()
        print("Disconnected.")
