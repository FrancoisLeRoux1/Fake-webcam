import cv2
import numpy as np
import pickle
import pyaudio
import struct
import math
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--folder", help="folder for the videos", type=str, default=None)
parser.add_argument("--freezes", help="use freezes",
                    action="store_true")
args = parser.parse_args()
folder, freezes = args.folder, args.freezes
switch_mode_max_delay_in_s = 0.5

if folder is None:
    video_dirs = [x for x in os.listdir('videos') if os.path.isdir(os.path.join('videos', x))]
    if len(video_dirs) == 0:
        raise Exception('No folder found for videos')
    else:
        folder = video_dirs[0]


def read_frames(file, video_folder):
    frames = []
    cap = cv2.VideoCapture(os.path.join('videos', video_folder, file))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened():
        print("Error opening video file")
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()
    return frames, frame_rate


video_files = [file for file in os.listdir(os.path.join('videos', folder))
               if file not in ['transitions_dict.p', '.DS_Store']]
frames, frame_rates = {}, {}

for file in video_files:
    mode_name = file.split('.')[0]
    frames[mode_name], frame_rates[mode_name] = read_frames(file, folder)
modes = list(frames.keys())
if 'normal' not in modes:
    raise Exception("No video named 'normal' found")
commands = {mode[0]: mode for mode in modes if mode != 'normal'}
print('Commands:')
for command, mode in commands.items():
    print(f"Press '{command}' to activate/deactivate '{mode}'")
print("Press 'v' to activate/deactivate voice detection")
if 'transitions_dict.p' not in os.listdir(os.path.join('videos', folder)):
    raise Exception(f"transitions_dict.p not found in {folder}, run 'python compute-transitions.py' first")
transitions_dict = pickle.load(open(os.path.join('videos', folder, 'transitions_dict.p'), 'rb'))

# region Voice detection

AMPLITUDE_THRESHOLD = 0.010
FORMAT = pyaudio.paInt16
SHORT_NORMALIZE = (1.0/32768.0)
CHANNELS = 1
RATE = 44100
INPUT_BLOCK_TIME = 0.025
INPUT_FRAMES_PER_BLOCK = int(RATE*INPUT_BLOCK_TIME)


def get_rms(block):
    count = len(block)/2
    format = "%dh" % count
    shorts = struct.unpack(format, block)

    sum_squares = 0.0
    for sample in shorts:
        n = sample * SHORT_NORMALIZE
        sum_squares += n*n
    return math.sqrt( sum_squares / count )


pa = pyaudio.PyAudio()

stream = pa.open(format=FORMAT,
                 channels=CHANNELS,
                 rate=RATE,
                 input=True,
                 frames_per_buffer=INPUT_FRAMES_PER_BLOCK)


def detect_voice():
    error_count = 0
    voice_detected = False

    try:
        block = stream.read(INPUT_FRAMES_PER_BLOCK, exception_on_overflow=False)
    except (IOError, e):
        error_count += 1
        print("(%d) Error recording: %s" % (error_count, e))

    amplitude = get_rms(block)
    if amplitude > AMPLITUDE_THRESHOLD:
        voice_detected = True
    return voice_detected

# endregion


def next_frame_index(i, mode, reverse):
    if i == len(frames[mode]) - 1:
        reverse = True
    if i == 0:
        reverse = False
    if not reverse:
        i += 1
    else:
        i -= 1
    return i, reverse


def change_mode(current_mode, toggled_mode, i, transition_freeze_duration_constant=1):
    if current_mode == toggled_mode:
        toggled_mode = 'normal'

    # Wait for the optimal frame to transition within acceptable window
    max_frames_delay = int(frame_rate * switch_mode_max_delay_in_s)
    global rev
    if rev:
        frames_to_wait = max_frames_delay-1 - transitions_dict[current_mode][toggled_mode][1][max(0, i+1 - max_frames_delay):i+1].argmin()
    else:
        frames_to_wait = transitions_dict[current_mode][toggled_mode][1][i:i + max_frames_delay].argmin()
    print(f'Wait {frames_to_wait} frames before transitioning')
    for _ in range(frames_to_wait):
        i, rev = next_frame_index(i, current_mode, rev)
        frame = frames[mode][i]
        cv2.imshow('Frame', frame)
        cv2.waitKey(int(1000 / frame_rate))

    new_i = transitions_dict[current_mode][toggled_mode][0][i]
    dist = transitions_dict[current_mode][toggled_mode][1][i]
    if freezes:
        freeze_duration = int(transition_freeze_duration_constant * dist)
        print(f'Froze for {freeze_duration} ms')
        cv2.waitKey(freeze_duration)
    print(f"Switched to '{toggled_mode}' mode")
    return new_i, toggled_mode, frame_rates[toggled_mode]


mode = "normal"
frame_rate = frame_rates[mode]
voice_detection = False
rev = False
i = 0
stop_talking_threshold = 10
quiet_count = 0
while True:
    frame = frames[mode][i]
    cv2.imshow('Webcam', frame)
    pressed_key = cv2.waitKey(int(1000/frame_rate)) & 0xFF
    if pressed_key == ord("q"):
        break
    elif pressed_key == ord("v"):
        voice_detection = not voice_detection
        print(f"Voice detection = {voice_detection}")
    for command, new_mode in commands.items():
        if pressed_key == ord(command):
            i, mode, frame_rate = change_mode(mode, new_mode, i)

    if voice_detection:
        if detect_voice():
            quiet_count = 0
            if mode != "talking":
                i, mode, frame_rate = change_mode(mode, "talking", i)
        else:
            if mode == "talking":
                quiet_count += 1
                if quiet_count > stop_talking_threshold:
                    quiet_count = 0
                    i, mode, frame_rate = change_mode(mode, "normal", i)
    # Random freezes
    if freezes:
        if np.random.randint(frame_rate * 10) == 1:
            nb_frames_freeze = int(np.random.uniform(0.2, 1.5) * frame_rate)
            print(f"Random freeze for {(nb_frames_freeze/frame_rate):.1f}s")
            for _ in range(nb_frames_freeze):
                cv2.waitKey(int(1000 / frame_rate))
                i, rev = next_frame_index(i, mode, rev)

    i, rev = next_frame_index(i, mode, rev)

# Closes all the frames
cv2.destroyAllWindows()


