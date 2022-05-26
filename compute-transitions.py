import cv2
import numpy as np
import pickle
from tqdm import tqdm
import os

def read_frames(file, video_folder):
    frames = []
    cap = cv2.VideoCapture(os.path.join('videos', video_folder, file))
    if not cap.isOpened():
        print("Error opening video  file")
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()
    return frames

list_video_folders = [f for f in os.listdir('videos') if os.path.isdir(os.path.join('videos', f))]

for video_folder in list_video_folders:
    if 'transitions_dict.p' not in os.listdir(os.path.join('videos', video_folder)):
        print(f'Computing transitions for {video_folder}')
        video_files = [file for file in os.listdir(os.path.join('videos', video_folder))
                       if file not in ['transitions_dict.p', '.DS_Store']]
        print('Reading videos')
        frames = {}
        for file in video_files:
            mode_name = file.split('.')[0]
            frames[mode_name] = read_frames(file, video_folder)
        modes = list(frames.keys())
        if 'normal' not in modes:
            raise Exception(f"No video named 'normal' found in folder {video_folder}")
        
        compression_ratio = 10
        height, width = frames["normal"][0].shape[:2]
        new_height, new_width = height // compression_ratio, width // compression_ratio, 

        def compress_img(img):
            return cv2.resize(img.mean(axis=2), (new_width, new_height))

    
        frames_compressed = {mode: np.array([compress_img(img) for img in frames[mode]]) for mode in modes}

        transitions_dict = {mode:{} for mode in modes}
        
        print('Computing optimal transitions, this may take a few minutes...')
        for i in range(len(modes)):
            for j in tqdm(range(i+1, len(modes))):
                mode_1, mode_2 = modes[i], modes[j]
                diff = np.expand_dims(frames_compressed[mode_1], axis=0) - np.expand_dims(frames_compressed[mode_2], axis=1)
                dists = np.linalg.norm(diff, axis=(2, 3))
                transitions_dict[mode_1][mode_2] = (dists.argmin(axis=0), dists.min(axis=0))
                transitions_dict[mode_2][mode_1] = (dists.argmin(axis=1), dists.min(axis=1))

        pickle.dump(transitions_dict, open(os.path.join('videos', video_folder, 'transitions_dict.p'), 'wb'))
        print(f'Done for {video_folder}')