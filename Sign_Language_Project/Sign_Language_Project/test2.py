import cv2
import tkinter as tk
from tkinter import *
from PIL import Image,ImageTk
from datetime import datetime
from PIL import ImageTk, Image
from tkinter import messagebox, filedialog
from function import *
import tkinter as tk
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


# Defining CreateWidgets() function to create necessary tkinter widgets
def createwidgets():
    root.feedlabel = Label(root, bg="steelblue", fg="white", text="Sign Language Project", font=('Comic Sans MS',20))
    root.feedlabel.grid(row=1, column=1, padx=10, pady=10, columnspan=2)

    root.cameraLabel = Label(root, bg="steelblue", borderwidth=3, relief="groove")
    root.cameraLabel.grid(row=2, column=1, padx=10, pady=10, columnspan=2)

    # root.saveLocationEntry = Entry(root, width=55, textvariable=destPath)
    # root.saveLocationEntry.grid(row=3, column=1, padx=10, pady=10)

    root.browseButton = Button(root, text="Speech", command=speech, bg="LIGHTBLUE", font=('Comic Sans MS',15), width=20)
    root.browseButton.grid(row=4, column=4, padx=10, pady=10)

    root.captureBTN = Button(root, text="Load Models", command=getframes, bg="LIGHTBLUE", font=('Comic Sans MS',15), width=20)
    root.captureBTN.grid(row=4, column=1, padx=10, pady=10)

    root.CAMBTN = Button(root, text="Predictions", command=prediction, bg="LIGHTBLUE", font=('Comic Sans MS',15), width=13)
    root.CAMBTN.grid(row=4, column=2)
    root.AboutBTN = Button(root, text="About", command=About, bg="LIGHTBLUE", font=('Comic Sans MS', 15),
                         width=13)
    root.AboutBTN.grid(row=1, column=6)
    root.HelpBTN = Button(root, text="Guestures", command=Help, bg="LIGHTBLUE", font=('Comic Sans MS', 15),
                         width=13)
    root.HelpBTN.grid(row=1, column=8)


    # root.openImageButton = Button(root, width=10, text="BROWSE", command=imageBrowse)
    # root.openImageButton.grid(row=3, column=5, padx=10, pady=10)

    # Calling ShowFeed() function
    #ShowFeed()

# Defining ShowFeed() function to display webcam feed in the cameraLabel;
def ShowFeed():
    # Capturing frame by frame
    ret, frame = root.cap.read()

    if ret:
        # Flipping the frame vertically
        frame = cv2.flip(frame, 1)

        # Displaying date and time on the feed
        cv2.putText(frame, datetime.now().strftime('%d/%m/%Y %H:%M:%S'), (20,30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,255))
        frame = cv2.rectangle(frame, (0, 40), (300, 400), 255, 2)
        frame = cv2.putText(frame, "Just Load Your Models  " + '', (75, 25), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            2, 255, 2)
        # Changing the frame color from BGR to RGB
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

        # Creating an image memory from the above frame exporting array interface
        videoImg = Image.fromarray(cv2image)

        # Creating object of PhotoImage() class to display the frame
        imgtk = ImageTk.PhotoImage(image = videoImg)

        # Configuring the label to display the frame
        root.cameraLabel.configure(image=imgtk)

        # Keeping a reference
        root.cameraLabel.imgtk = imgtk

        # Calling the function after 10 milliseconds
        root.cameraLabel.after(10, ShowFeed)
    else:
        # Configuring the label to display the frame
        root.cameraLabel.configure(image='')
#speech button
def speech(sentence):
    text_speech = pyttsx3.init()
    text_speech.say(sentence)
    text_speech.runAndWait()


#
# Defining Capture() to capture and save the image and display the image in the imageLabel
#get frames button
def getframes():
    # ShowFeed()
    json_file = open("model.json", "r")
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights("model.h5")
    print("your model is successfully loaded: Please Move Forward ")





#prediction button
def prediction():
    text_speech = pyttsx3.init()
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=2)
    offset = 40
    imgsize = 300
    colors = []
    ###############################
    ret, frame = root.cap.read()

        # Calling the function after 10 milliseconds


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
    comp = ['90']
    comp1 = ['100']
    x = [1, 2, 3, 4]
    accuracy = []
    predictions = []
    threshold = 0.8
    fps_start_time = 0
    fps = 0



    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("https://192.168.43.41:8080/video")
    # Set mediapipe model
    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:

        while True:
            success, img = root.cap.read()
            imgoutput = img.copy()
            fps_end_time = time.time()
            time_diff = fps_end_time - fps_start_time
            fps = 1 / (time_diff)
            fps_start_time = fps_end_time
            fps_text = "FPS: {:.2f}".format(fps)
            hands1, img = detector.findHands(img)

            # Read feed
            ret, frame = root.cap.read()
            ret, frame2 = root.cap.read()

            # Make detections
            cropframe = frame[40:400, 0:300]
            # print(frame.shape)
            frame = cv2.rectangle(frame, (0, 40), (300, 400), 255, 2)
            frame = cv2.putText(frame, "Active Region  " + ''.join(fps_text), (75, 25), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                2, 255, 2)
            image, results = mediapipe_detection(cropframe, hands)
            # print(results)

            # Draw landmarks
            draw_styled_landmarks(image, results)
            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            try:
                t = 0
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
                        accuracy = accuracy[-1:]
                        # accuracy1=accuracy
                        # accuracy2=math.ceil(accuracy1)

                    # Viz probabilities
                    frame = prob_viz(res, actions, frame, colors, threshold)
                    cv2.imshow("imgCrop", imgCrop)
                    # cv2.imshow("imgwhite", imgwhite)
                    if len(sentence) > 1 and (accuracy>comp or accuracy==comp1):
                        frame = cv2.rectangle(frame, (0, 40), (700, 500), 0, 2)
                        frame2 = cv2.rectangle(frame, (300, 100), (700, 40), 0, 2)
                    cv2.rectangle(frame, (0, 0), (800, 40), (245, 117, 16), -2)
                    cv2.putText(frame, " " + ' '.join(sentence) + ' A: ' + "".join(accuracy), (3, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    # text_speech.say(sentence)
                    # text_speech.runAndWait()
                    # cv2.putText(frame, (accuracy), (30, 50),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)






            except Exception as e:
                # print(e)
                pass

                # text_speech.runAndWait()

                # if t== 2:
                #     goto .myLabel
            # else:
            #     cv2.rectangle(frame, (0, 0), (800, 40), (245, 117, 16), -2)
            #     cv2.putText(frame, "please show a valid sign: -", (3, 30),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # cv2.rectangle(frame, (300, 400), (700, 40), (255, 117, 205), -2)
            # f1 = cv2.rectangle(frame, (300, 100), (700, 40), (134, 117, 255), -2)
            # # cv2.imshow('Rectangle', image1)
            # root.cameraLabel.after(10, ShowFeed)
        # else:
        #     # Configuring the label to display the frame
        #     root.cameraLabel.configure(image='')

            # Show to screen
            cv2.imshow('OpenCV Feed', frame)

            # Break gracefully
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        # speech(sentence)

    # Stopping the camera using release() method of cv2.VideoCapture()
def About():
    root1 = tk.Toplevel(root)
    root1.title("About")

    root1.geometry("950x900")

    # img = cv2.imread("Pictures/sir.jpg", 1)
    # # img = cv2.resize(img, (300, 300))
    # cv2.imwrite("Pictures/sir.png", img)
    # return

    tx = tk.Label(root1)
    tx.place(x=200, y=20)
    tx.config(text="Efforts By", fg="red", font=("Courier", 30, "bold"))
    tx1 = tk.Label(root1)
    tx1.place(x=200, y=90)
    tx1.config(text="Muhammad Touseef(192-101-002)", fg="red", font=("Courier", 30, "bold"))
    tx2 = tk.Label(root1)
    tx2.place(x=200, y=160)
    tx2.config(text="Saddam Ahmed Nisar(192-101-024)", fg="red", font=("Courier", 30, "bold"))
    tx3 = tk.Label(root1)
    tx3.place(x=200, y=250)
    tx3.config(text="Guided By", fg="Blue", font=("Courier", 30, "bold"))
    tx4 = tk.Label(root1)
    tx4.place(x=200, y=340)
    tx4.config(text="Dr. Usman Gulzari", fg="Blue", font=("Courier", 30, "bold"))

    # photo1 = tk.PhotoImage(file='Pictures/saddam.jpg')
    # w1 = tk.Label(root1, image=photo1)
    # w1.place(x=20, y=105)
    # tx6 = tk.Label(root1)
    # tx6.place(x=20, y=250)
    # tx6.config(text="RC\nIIT2016141", font=("Courier", 15, "bold"))
    #
    # photo2 = tk.PhotoImage(file='Pictures/saddam.jpg')
    # w2 = tk.Label(root1, image=photo2)
    # root1.w2.place(x=200, y=105)
    # root1.tx2 = tk.Label(root1)
    # root1.tx2.place(x=200, y=250)
    # root1.tx2.config(text="Nitin\nIIT2016132", font=("Courier", 15, "bold"))
    #
    # photo3 = tk.PhotoImage(file='Pictures/luv.png')
    # w3 = tk.Label(root1, image=photo3)
    # w3.place(x=380, y=105)
    # tx3 = tk.Label(root1)
    # tx3.place(x=380, y=250)
    # tx3.config(text="Luv\nIIT2016085", font=("Courier", 15, "bold"))
    # #
    # photo4 = tk.PhotoImage(file='Pictures/sheldon.png')
    # w4 = tk.Label(root1, image=photo4)
    # w4.place(x=560, y=105)
    # tx4 = tk.Label(root1)
    # tx4.place(x=560, y=250)
    # tx4.config(text="Sheldon\nIIT2016137", font=("Courier", 15, "bold"))
    #
    # photo5 = tk.PhotoImage(file='Pictures/sid.png')
    # w5 = tk.Label(root1, image=photo5)
    # w5.place(x=740, y=105)
    # tx5 = tk.Label(root1)
    # tx5.place(x=740, y=250)
    # tx5.config(text="Siddhant\nIIT2016069", font=("Courier", 15, "bold"))
    #
    # tx7 = tk.Label(root1)
    # tx7.place(x=170, y=360)
    # tx7.config(text="Under the supervision of", fg="red", font=("Courier", 30, "bold"))
    #
    # photo6 = tk.PhotoImage(file='Pictures/sir.png')
    # w6 = tk.Label(root1, image=photo6)
    # w6.place(x=350, y=420)
    # tx6 = tk.Label(root1)
    # tx6.place(x=230, y=670)
    # tx6.config(text="Dr. Vrijendra Singh", font=("Courier", 30, "bold"))


def Help():
    # Presenting user with a pop-up for directory selection. initialdir argument is optional
    # Retrieving the user-input destination directory and storing it in destinationDirectory
    # Setting the initialdir argument is optional. SET IT TO YOUR DIRECTORY PATH
    # Create an instance of tkinter window
    # Create an instance of tkinter window
    ShowFeed()

    # Creating object of tk class
root = tk.Tk()

# Creating object of class VideoCapture with webcam index
root.cap = cv2.VideoCapture(0)

# Setting width and height
width, height = 640, 480
root.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
root.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Setting the title, window size, background color and disabling the resizing property
root.title("Sign Language")
root.geometry("1040x700")
root.resizable(True, True)
root.configure(background = "sky blue")

# Creating tkinter variables
destPath = StringVar()
imagePath = StringVar()

createwidgets()
root.mainloop()
