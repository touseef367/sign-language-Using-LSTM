# import cv2
# import mediapipe as mp
# mp_drawing=mp.solutions.drawing_utils
# mp_drawing_style=mp.solutions.drawing_styles
# mphands=mp.solutions.hands
# cap=cv2.VideoCapture(0)
# hands=mphands.Hands()
# while True:
#     data,image=cap.read()
#     image=cv2.cvtColor(cv2.flip(image,1),cv2.COLOR_BGR2RGB)
#     result=hands.process(image)
#     image=image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#     if result.multi_hand_landmarks:
#         for hand_landmarks in result.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(image,hand_landmarks.HAND_CONNECTIONS)
#
#     cv2.imshow('hand tracker',image)
#     cv2.waitKey(1)
#

# Install the Libraries

import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
cTime = 0
pTime = 0
# Function Start
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            for id, lm in enumerate(handlms.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                # if id == 5:
                cv2.circle(img, (cx, cy), 15, (139, 0, 0), cv2.FILLED)

            mpDraw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)
    # Time and FPS Calculation

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (139, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)