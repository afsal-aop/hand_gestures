import time
import cv2 as cv
import mediapipe as mp
from tensorflow import keras
import numpy as np

def preprocess(x, y):
    joined_arr = np.array(x + y)
    return joined_arr

def findLabel(prediction):
    prediction = prediction[0]
    dominant_pred = np.argmax(prediction)
    label = class_label[dominant_pred]
    prob = round(prediction[dominant_pred] * 100, 2)
    return label, prob

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

class_label = ['Hii', 'peace', 'rock', 'well done']
gesture_model = keras.models.load_model('gesture_model.h5')

ctime = 0
ptime = 0
cam0 = cv.VideoCapture(0)

while True:
    success, frame = cam0.read()

    if success:
        frame = cv.flip(frame, 1)
        h, w, c = frame.shape
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                xs = []
                ys = []
                mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
                for lm in handLms.landmark:
                    xs.append(lm.x)
                    ys.append(lm.y)
                processed_instance = preprocess(xs, ys)
                prediction = gesture_model.predict(processed_instance.reshape(1, 42))
                label, prob = findLabel(prediction)
                cv.putText(frame, f'Detection: {label} [{prob}%]', (7, 120), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        ctime = time.time()
        fps = int(1 / (ctime - ptime)) if (ctime - ptime) > 0 else 0
        ptime = ctime
        cv.putText(frame, 'fps : ' + str(fps), (7, 30), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
        cv.imshow('frame', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

cam0.release()
cv.destroyAllWindows()
