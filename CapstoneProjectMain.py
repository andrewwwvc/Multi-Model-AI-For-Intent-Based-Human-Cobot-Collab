"""
Capstone Project - UR Robot Voice Command Interface
Step 1: Press SPACE to record a voice command
Step 2: spaCy matches the command semantically to a register value
Step 3: Write the register value to the UR robot via RTDE
        and poll until the robot finishes or reports an error

Install:
    python3.11 -m pip install speechrecognition pyaudio keyboard spacy ur-rtde
    python3.11 -m spacy download en_core_web_md

Run:
    sudo python3.11 CapstoneProject.py
"""

import threading
import time
import speech_recognition as sr
import keyboard
import spacy
import rtde_receive
import rtde_io

# -- Configuration -------------------------------------------------------------
ROBOT_IP = "192.168.1.112"      # Your robot's IP
ENERGY_THRESHOLD = 300          # Mic sensitivity
SIMILARITY_THRESHOLD = 0.70     # Minimum NLP match score (0.0 - 1.0)
POLL_RATE = 0.5                 # Seconds between register polls
MIC_NAME = "fifine"             # Partial name of your external microphone (case insensitive)

# -- Register indices ----------------------------------------------------------
REG_COMMAND = 18
REG_STATUS = 19
ERROR_PART_MISSING = 99
STATUS_DONE = 1

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
        "place the emergency light"
        "red light only",
    ],
    2: [
        "warning", "yellow", "caution",
        "place the yellow part",
        "place the yellow light",
        "assemble the yellow light",
        "place the warning light"
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
        "place the start light",
        "place the power light",
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

CANCEL_DOCS = {
    word: nlp(word) for word in ANCHORS.get(0, [])
}

print("spaCy model loaded.\n")

# -- Microphone Setup ----------------------------------------------------------
def find_microphone_index(name_fragment):
    """
    Search available microphones for one whose name contains name_fragment.
    Returns the device index, or None if not found.
    """
    mic_list = sr.Microphone.list_microphone_names()
    print("Available microphones:")
    for i, mic in enumerate(mic_list):
        print(f"  [{i}] {mic}")
    
    matches = [i for i, mic in enumerate(mic_list) if name_fragment.lower() in mic.lower()]
    if matches:
        # Use the last match to avoid virtual/duplicate entries
        i = matches[-1]
        print(f"\nUsing microphone [{i}]: {mic_list[i]}\n")
        return i
    
    print(f"\nWarning: '{name_fragment}' microphone not found. Using default mic.\n")
    return None

MIC_INDEX = find_microphone_index(MIC_NAME)

# -- Speech Recognition Setup --------------------------------------------------
recogniser = sr.Recognizer()
recogniser.energy_threshold = ENERGY_THRESHOLD
recogniser.dynamic_energy_threshold = False


# -- RTDE Helpers --------------------------------------------------------------
def read_reg(rtde_r, index):
    return int(rtde_r.getOutputIntRegister(index))


def write_reg(rtde_io_, index, value):
    rtde_io_.setInputIntRegister(index, value)
    print(f"Register[{index}] -> {value}")


def clear_registers(rtde_io_):
    write_reg(rtde_io_, REG_COMMAND, 0)
    write_reg(rtde_io_, REG_STATUS, 0)


# -- Step 1: Speech-to-Text ----------------------------------------------------
def listen_for_command():
    """
    Wait for SPACE to start recording, SPACE again to stop.
    Uses the Fifine microphone if found, otherwise falls back to default.
    Returns the recognised text, or None if nothing was understood.
    """
    print("Press SPACE to start recording ...")
    keyboard.wait("space")
    print("Recording ... press SPACE to stop.")

    stop_event = threading.Event()
    hook = keyboard.on_press_key("space", lambda _: stop_event.set())

    frames = []

    with sr.Microphone(device_index=MIC_INDEX) as source:
        recogniser.adjust_for_ambient_noise(source, duration=0.5)
        sample_rate = source.SAMPLE_RATE
        sample_width = source.SAMPLE_WIDTH

        while not stop_event.is_set():
            try:
                chunk = recogniser.record(source, duration=0.5)
                frames.append(chunk.get_raw_data())
            except Exception as e:
                print(f"Audio capture error: {e}")
                break

    keyboard.unhook(hook)

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
    Also prints how long the NLP matching took.
    """
    nlp_start = time.perf_counter()

    user_input_clean = user_input.lower().strip()
    doc = nlp(user_input_clean)

    best_score = 0.0
    best_anchor = ""
    best_reg = 0

    # Check regular anchors (registers 1-4)
    for phrase, (reg_val, anchor_doc) in ANCHOR_DOCS.items():
        if reg_val == 0:
            continue
        score = doc.similarity(anchor_doc)
        if score > best_score and score >= SIMILARITY_THRESHOLD:
            best_score = score
            best_anchor = phrase
            best_reg = reg_val

    # Check cancel anchors separately
    for word, cancel_doc in CANCEL_DOCS.items():
        score = doc.similarity(cancel_doc)
        if score > best_score and score >= SIMILARITY_THRESHOLD:
            best_score = score
            best_anchor = "cancel"
            best_reg = 0

    nlp_elapsed = time.perf_counter() - nlp_start

    if best_anchor:
        print(f"NLP match time: {nlp_elapsed:.4f} seconds")
    else:
        print(f"NLP time: {nlp_elapsed:.4f} seconds (no match found)")

    return best_reg, best_anchor, best_score


# -- Step 3: Robot Sync --------------------------------------------------------
def wait_for_robot(rtde_r, rtde_io_):
    """
    Simply notifies the user to wait for the robot to finish
    before speaking the next command.
    """
    print("\nWait for robot to finish task before speaking next command.\n")
    return True


def send_command(rtde_r, rtde_io_, reg_value):
    """
    Write command to register 18 and wait for robot to finish.
    Returns True on success, False on error/busy.
    """
    current_command = read_reg(rtde_r, REG_COMMAND)
    current_status = read_reg(rtde_r, REG_STATUS)

    if current_command != 0 or current_status != 0:
        print(
            f"Robot is busy "
            f"(command={current_command}, status={current_status}). Command not sent."
        )
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

    # Clear stale values at startup
    clear_registers(rtde_io_)
    print("Registers cleared.\n")

    try:
        print("Press SPACE to record a command. (Ctrl+C to quit)\n")
        print("Example commands:")
        print("  'place the red light'      -> register 1")
        print("  'place the yellow light'   -> register 2")
        print("  'place the green light'    -> register 3")
        print("  'run the full sequence'    -> register 4")
        print("  'cancel'                   -> reset registers\n")

        while True:
            user_text = listen_for_command()

            if user_text is None:
                print("Nothing heard - try again.\n")
                continue

            user_text_clean = user_text.lower().strip()

            if user_text_clean in ("quit", "exit", "stop listening"):
                print("Exit command heard. Goodbye!")
                break

            # NLP matching with timer
            reg_value, matched_anchor, score = get_best_match(user_text_clean)

            # Handle cancel/reset
            if reg_value == 0 and matched_anchor == "cancel":
                print(f"Cancel command matched: '{matched_anchor}' (score={score:.2f})")
                clear_registers(rtde_io_)
                print("Registers reset. Ready for next command.\n")
                continue

            # No valid match
            if matched_anchor == "":
                print(
                    f"No confident match for '{user_text}' "
                    f"(below {SIMILARITY_THRESHOLD:.0%} threshold). Try again.\n"
                )
                continue

            print(
                f"Matched '{user_text}' -> anchor='{matched_anchor}' "
                f"(score={score:.2f}) -> register {reg_value}"
            )

            success = send_command(rtde_r, rtde_io_, reg_value)

            if success:
                print("Work order complete. Ready for next command.\n")
            else:
                print("Command failed or robot busy. Please check the robot.\n")

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        try:
            clear_registers(rtde_io_)
        except Exception:
            pass
        rtde_r.disconnect()
        rtde_io_.disconnect()
        print("Disconnected.")
