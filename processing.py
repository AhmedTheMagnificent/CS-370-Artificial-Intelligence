import os
import cv2
import mediapipe as mp
import pickle

mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDrawing = mp.solutions.drawing_utils

dataDir = r"A:\archive (1) 2\ASL_Dataset/Train"
data = []
labels = []

for i in sorted(os.listdir(dataDir)):
    for j in os.listdir(os.path.join(dataDir, i))[:1000]:
        dataAux = []
        img = cv2.imread(os.path.join(dataDir, i, j))
        if img is None:
            continue

        imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRgb)
        if results.multi_hand_landmarks:
            for handLandmarks in results.multi_hand_landmarks:
                mpDrawing.draw_landmarks(imgRgb, handLandmarks, mpHands.HAND_CONNECTIONS)
                for landmark in handLandmarks.landmark:
                    x = landmark.x
                    y = landmark.y
                    dataAux.extend([x, y])
            if len(dataAux) == 42:
                data.append(dataAux)
                labels.append(i)

        cv2.imshow("image", imgRgb)
        cv2.waitKey(1)

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

hands.close()
cv2.destroyAllWindows()
