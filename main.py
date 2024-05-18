import cv2
import mediapipe as mp
import numpy as np
import pickle

model_to_load = 'RandomForest'  # Specify the model name you want to load

with open(f'{model_to_load}_model.pkl', 'rb') as f:
    model = pickle.load(f)

mpHands = mp.solutions.hands
mpDrawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mpHands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frameRgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frameRgb = cv2.flip(frameRgb, 1)
        frameRgb.flags.writeable = False
        results = hands.process(frameRgb)
        frameRgb.flags.writeable = True 
        frameRgb = cv2.cvtColor(frameRgb, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for handLandmarks in results.multi_hand_landmarks:
                mpDrawing.draw_landmarks(
                    frameRgb,
                    handLandmarks,
                    mpHands.HAND_CONNECTIONS,
                    mpDrawing.DrawingSpec(color=(28, 255, 3), thickness=5, circle_radius=10),
                    mpDrawing.DrawingSpec(color=(236, 255, 3), thickness=5, circle_radius=10)
                )

            dataAux = []
            x_ = []
            y_ = []

            for handLandmarks in results.multi_hand_landmarks:
                for i in range(len(handLandmarks.landmark)):
                    x = handLandmarks.landmark[i].x
                    y = handLandmarks.landmark[i].y
                    dataAux.append(x)
                    dataAux.append(y)
                    x_.append(x)
                    y_.append(y)

            x1 = int(min(x_) * frame.shape[1]) - 10
            y1 = int(min(y_) * frame.shape[0]) - 10
            x2 = int(max(x_) * frame.shape[1]) - 10
            y2 = int(max(y_) * frame.shape[0]) - 10

            prediction = model.predict([np.array(dataAux)[:42]])[0]

            cv2.rectangle(frameRgb, (x1, y1 - 10), (x2, y2), (255, 99, 173), 6)
            cv2.putText(frameRgb, prediction, (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 5, (255, 0, 0), 5, cv2.LINE_AA)

        cv2.imshow('frame', frameRgb)  
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
