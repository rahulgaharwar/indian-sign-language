import cv2
import mediapipe as mp
import copy
import itertools
from tensorflow import keras
import numpy as np
import pandas as pd
import string
import tkinter as tk
from tkinter import ttk
from collections import Counter
from PIL import Image, ImageTk
import threading

# Load the saved model from file
model = keras.models.load_model("C:\\Users\\rahul\\Indian-Sign-Language-Detection\\model.h5")

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

alphabet =  ['1','2','3','4','5','6','7','8','9']
alphabet += list(string.ascii_uppercase)

# Initialize text-to-speech engine
import pyttsx3
engine = pyttsx3.init()

# Global variables for detection
word = ""
detected_letters = []
detecting = True

def calc_landmark_list(image, landmarks):
    """Calculate the landmark points for the given image."""
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    """Normalize the landmark points."""
    temp_landmark_list = copy.deepcopy(landmark_list)

    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] -= base_x
        temp_landmark_list[index][1] -= base_y

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def clear_word():
    """Clear the detected word."""
    global word
    word = ""
    label_word.config(text=word)

def speak_word():
    """Speak the detected word."""
    global word
    engine.say(word)
    engine.runAndWait()

def update_detected_letter():
    """Update the detected letter every 10 seconds."""
    global detected_letters, word, detecting
    if detected_letters:
        most_common_letter = Counter(detected_letters).most_common(1)[0][0]
        word += most_common_letter
        label_word.config(text=word)
        detected_letters = []
    detecting = True
    root.after(10000, update_detected_letter)

def update_frame():
    """Update the webcam frame and detect hand gestures."""
    global detected_letters, detecting, current_letter
    success, image = cap.read()
    image = cv2.flip(image, 1)
    if not success:
        root.after(10, update_frame)
        return

    image.flags.writeable = False
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    image.flags.writeable = True
    debug_image = copy.deepcopy(image)

    if results.multi_hand_landmarks and detecting:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            landmark_list = calc_landmark_list(debug_image, hand_landmarks)
            pre_processed_landmark_list = pre_process_landmark(landmark_list)
            mp_drawing.draw_landmarks(
                debug_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            df = pd.DataFrame(pre_processed_landmark_list).transpose()
            predictions = model.predict(df, verbose=0)
            predicted_classes = np.argmax(predictions, axis=1)
            label = alphabet[predicted_classes[0]]

            detected_letters.append(label)
            label_letter.config(text="Detected Letter: " + label)
            current_letter = label

    # Display the current detected letter on the image
    if detected_letters:
        current_letter = detected_letters[-1]
        cv2.putText(debug_image, current_letter, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

    # Convert the image to RGB for PIL
    img_rgb = cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)

    root.after(10, update_frame)

# Initialize the GUI
root = tk.Tk()
root.title("Indian Sign Language Detector")

frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

label_letter = ttk.Label(frame, text="Detected Letter: ", font=("Helvetica", 16))
label_letter.grid(row=0, column=0, padx=5, pady=5)

label_word = ttk.Label(frame, text="", font=("Helvetica", 16))
label_word.grid(row=1, column=0, padx=5, pady=5)

button_clear = ttk.Button(frame, text="Clear", command=clear_word)
button_clear.grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)

button_speak = ttk.Button(frame, text="Speak", command=speak_word)
button_speak.grid(row=2, column=0, padx=5, pady=5, sticky=tk.E)

# Display the webcam feed in the GUI
lmain = tk.Label(frame)
lmain.grid(row=3, column=0, padx=5, pady=5)

# For webcam input
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    root.after(10, update_frame)
    root.after(10000, update_detected_letter)  # Start the 10-second timer for updating detected letter
    root.mainloop()