import cv2
import mediapipe as mp

paint = mp.solutions.drawing_utils
hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as dt:
    while cap.isOpened():

        success, image = cap.read()
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = dt.process(image)
        image.flags.writeable = True

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                paint.draw_landmarks(image, hand_landmarks, hands.HAND_CONNECTIONS)

        cv2.imshow('Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
