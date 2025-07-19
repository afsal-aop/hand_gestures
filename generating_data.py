import mediapipe as mp
import cv2 as cv
import numpy as np
import os

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

data_points = []
data_labels = []

total_frames = 2000  # Set how many frames to capture per video
video_root = 'train_videos/'

# Ensure the output data directory exists
os.makedirs('data', exist_ok=True)

for video_idx, video in enumerate(os.listdir(video_root)):
    if not video.lower().endswith(('.mp4', '.avi', '.mov')):
        continue  # Skip non-video files

    cam = cv.VideoCapture(os.path.join(video_root, video))
    frame_count = 0
    while True:
        success, frame = cam.read()
        if not success:
            break
        frame = cv.flip(frame, 1)
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        processed = hands.process(frame_rgb)

        if processed.multi_hand_landmarks:
            for handLms in processed.multi_hand_landmarks:
                mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
                xs = [lm.x for lm in handLms.landmark]
                ys = [lm.y for lm in handLms.landmark]
                data_points.append(xs + ys)
                data_labels.append(video_idx)
                frame_count += 1

        cv.imshow('frame', frame)
        if cv.waitKey(1) & 0xFF == ord('q') or frame_count == total_frames:
            break

    print(video.split('.')[0], video_idx, frame_count)
    cam.release()
    cv.destroyAllWindows()

print(data_labels)
np.save('data/data', data_points)
np.save('data/labels', data_labels)
