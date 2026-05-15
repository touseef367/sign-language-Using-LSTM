import cv2
#from goto import goto,label
from function import *
from keras.utils import to_categorical
from keras.models import model_from_json
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
import time
json_file = open("model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model.h5")
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
from cvzone.ClassificationModule import Classifier
import pyttsx3

text_speech=pyttsx3.init()
cap=cv2.VideoCapture(0)
detector=HandDetector(maxHands=2)
offset=40
imgsize = 300
colors = []
###############################

#######################################
for i in range(0, 20):
    colors.append((245, 117, 16))
print(len(colors))


def prob_viz(res, actions, input_frame, colors, threshold):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)

    return output_frame


# 1. New detection variables
sequence = []
sentence = []
comp=['90']
comp1=['100']
x=[1,2,3,4]
accuracy = []
predictions = []
threshold = 0.8
fps_start_time=0
fps=0

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("https://192.168.43.41:8080/video")
# Set mediapipe model
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

    while True:
        success, img = cap.read()
        imgoutput = img.copy()
        fps_end_time=time.time()
        time_diff=fps_end_time-fps_start_time
        fps=1/(time_diff)
        fps_start_time=fps_end_time
        fps_text="FPS: {:.2f}".format(fps)
        hands1, img = detector.findHands(img)

        # Read feed
        ret, frame = cap.read()
        ret, frame2 = cap.read()

        # Make detections
        cropframe = frame[40:400, 0:300]
        # print(frame.shape)
        frame = cv2.rectangle(frame, (0, 40), (300, 400), 255, 2)
        frame=cv2.putText(frame,"Active Region  "+ ''.join(fps_text),(75,25),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,255,2)
        image, results = mediapipe_detection(cropframe, hands)
        # print(results)

        # Draw landmarks
        # draw_styled_landmarks(image, results)
        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        try:
            t=0
            if hands1:

                hand = hands1[0]
                x, y, w, h = hand['bbox']
                # imgwhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
                imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
                imgCropshape = imgCrop.shape
                aspectratio = h / w

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))

                # 3. Viz logic
                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                                accuracy.append(str(res[np.argmax(res)] * 100))

                        else:
                            sentence.append(actions[np.argmax(res)])
                            accuracy.append(str(res[np.argmax(res)] * 100))

                if len(sentence) > 1:
                    sentence = sentence[-1:]
                    accuracy = accuracy[-1:0]
                    accuracy1=accuracy
                    accuracy2=math.ceil(accuracy1)


                # Viz probabilities
                # frame = prob_viz(res, actions, frame, colors,threshold)
                cv2.imshow("imgCrop", imgCrop)
                # cv2.imshow("imgwhite", imgwhite)
                if len(sentence) > 1:
                    frame = cv2.rectangle(frame, (0, 40), (300, 400), 0, 2)
                    frame2 = cv2.rectangle(frame, (300, 100), (700, 40), 0, 2)

                cv2.rectangle(frame, (0, 0), (800, 40), (245, 117, 16), -2)
                # cv2.putText(frame, "Output: -" + ' '.join(sentence) + ''.join(accuracy), (3, 30),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                if (accuracy > comp or accuracy == comp1):
                    cv2.rectangle(frame, (0, 0), (800, 40), (255, 0, 255), -2)
                    cv2.putText(frame, "" + ' '.join(sentence) + " A: " + ''.join(accuracy), (3, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    print("A ", type(accuracy))
                    # cv2.rectangle(frame, (300, 400), (700, 40), (25, 117, 205), -2)
                    # f1 = cv2.rectangle(frame, (300, 100), (700, 40), (134, 117, 255), -2)



                    # text_speech.say(sentence)
                    # text_speech.runAndWait()

                    # text_speech.runAndWait()

                    # if t== 2:
                    #     goto .myLabel
                else:
                 cv2.rectangle(frame, (0, 0), (800, 40), (245, 117, 16), -2)
                 cv2.putText(frame, "please show a valid sign: -", (3, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                # cv2.rectangle(frame, (300, 400), (700, 40), (255, 117, 205), -2)
                # f1 = cv2.rectangle(frame, (300, 100), (700, 40), (134, 117, 255), -2)
                # # cv2.imshow('Rectangle', image1)











        except Exception as e:
            # print(e)
            pass

        # Show to screen
        cv2.imshow('OpenCV Feed', frame)



        # Break gracefully
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
