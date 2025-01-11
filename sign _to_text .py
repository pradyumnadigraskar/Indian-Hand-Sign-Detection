
#update version 

# Imported necessary library 
import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
from gtts import gTTS
import os
import pyttsx3
engine = pyttsx3.init()

#speech to text function
def speak_text(text):
    """
    Speak the given text using pyttsx3.
    """
    engine.say(text)
    engine.runAndWait()

speak_button_position = None
# Load the trained model
model_dict = pickle.load(open('three.p', 'rb'))
model = model_dict['model']
scaler = model_dict['scaler']

cap = cv2.VideoCapture(0)

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7)

# Predefined dictionary of words for suggestions
dictionary = ["HELLO", "HELL", "HELMET", "HELP", "HAPPY", "HAND", "PYTHON"]

# Sentence tracking
detected_sentence = []  # List to store detected characters
sentence_field = ""  # Variable to hold the sentence as a string
final_field = ""  # Variable to hold the final accepted sentence
word_suggestions = []  # List to hold word suggestions

last_predicted_character = None
stable_predictions = []
cooldown_time = 0.2
last_detection_time = time.time()

# Button positions for suggestions, clear, and space buttons
button_positions = []
clear_button_position = None
space_button_position = None

def draw_interface(frame, sentence, final_sentence, suggestions):
    """
    Draw the interface with the camera feed, sentence, final sentence, and suggestions.
    """
    global button_positions, clear_button_position, space_button_position, speak_button_position
    button_positions.clear()

    # Create a larger frame to hold the camera feed and suggestion panel
    panel_height = 200
    display_frame = np.zeros((frame.shape[0] + panel_height, frame.shape[1], 3), dtype=np.uint8)

    # Copy the camera feed to the top portion
    display_frame[:frame.shape[0], :] = frame

    # Draw the sentence area
    cv2.rectangle(display_frame, (0, frame.shape[0]), (frame.shape[1], frame.shape[0] + panel_height), (50, 50, 50), -1)
    cv2.putText(display_frame, "Sentence: " + sentence, (10, frame.shape[0] + 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2)
    cv2.putText(display_frame, "Final: " + final_sentence, (10, frame.shape[0] + 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 255), 2)

    # Draw suggestion buttons
    x_start = 10
    y_start = frame.shape[0] + 110
    button_width = 150
    button_height = 50
    spacing = 10

    for i, suggestion in enumerate(suggestions[:5]):  # Limit to 5 suggestions
        x1 = x_start + i * (button_width + spacing)
        y1 = y_start
        x2 = x1 + button_width
        y2 = y1 + button_height

        button_positions.append((x1, y1, x2, y2, suggestion))

        # Draw the button rectangle
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), -1)
        # Add the suggestion text
        cv2.putText(display_frame, suggestion, (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Draw Clear and Space buttons
    clear_button_position = (10, frame.shape[0] + panel_height - 60, 160, frame.shape[0] + panel_height - 10)
    cv2.rectangle(display_frame, clear_button_position[:2], clear_button_position[2:], (0, 0, 255), -1)
    cv2.putText(display_frame, "Clear", (clear_button_position[0] + 30, clear_button_position[1] + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    space_button_position = (180, frame.shape[0] + panel_height - 60, 330, frame.shape[0] + panel_height - 10)
    cv2.rectangle(display_frame, space_button_position[:2], space_button_position[2:], (255, 255, 0), -1)
    cv2.putText(display_frame, "Space", (space_button_position[0] + 20, space_button_position[1] + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Draw Speak button
    speak_button_position = (350, frame.shape[0] + panel_height - 60, 500, frame.shape[0] + panel_height - 10)
    cv2.rectangle(display_frame, speak_button_position[:2], speak_button_position[2:], (0, 128, 255), -1)
    cv2.putText(display_frame, "Speak", (speak_button_position[0] + 30, speak_button_position[1] + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    return display_frame

def on_mouse_click(event, x, y, flags, param):
    global detected_sentence, sentence_field, final_field
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check for suggestion button clicks
        for button in button_positions:
            x1, y1, x2, y2, suggestion = button
            if x1 <= x <= x2 and y1 <= y <= y2:
                final_field += suggestion + " "  # Add the clicked suggestion to the final sentence
                sentence_field = ""  # Clear the sentence field after suggestion is selected
                detected_sentence = []  # Clear the detected sentence as well
                return

        # Check for Clear button click
        if clear_button_position[0] <= x <= clear_button_position[2] and clear_button_position[1] <= y <= clear_button_position[3]:
            detected_sentence = []  # Clear character-based sentence
            sentence_field = ""  # Clear the current sentence
            final_field = ""  # Clear the final sentence
            return

        # Check for Space button click
        if space_button_position[0] <= x <= space_button_position[2] and space_button_position[1] <= y <= space_button_position[3]:
            if sentence_field.upper() not in dictionary:
                # If the current sentence does not match any word in the dictionary,
                # add it to the final field as individual characters.
                final_field += sentence_field + " "
            detected_sentence.append(" ")
            sentence_field = ''.join(map(str, detected_sentence))  # Add space to the sentence
            return

        # Check for Speak button click
        if speak_button_position[0] <= x <= speak_button_position[2] and speak_button_position[1] <= y <= speak_button_position[3]:
            if final_field:
                speak_text(final_field)  # Convert the final sentence to speech
            return



cv2.namedWindow("frame")
cv2.setMouseCallback("frame", on_mouse_click)

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Prepare the data for both hands
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))

        while len(data_aux) < 126:
            data_aux.extend([0, 0])  # Pad with zeros if only one hand was detected

        data_input = np.asarray(data_aux).reshape(1, -1)
        data_input_scaled = scaler.transform(data_input)

        prediction = model.predict(data_input_scaled)
        predicted_character = prediction[0]

        if time.time() - last_detection_time >= cooldown_time:
            stable_predictions.append(predicted_character)

            if len(stable_predictions) >= 3:  # Check if prediction is stable
                if stable_predictions[0] == stable_predictions[1] == stable_predictions[2]:
                    if predicted_character != last_predicted_character:
                        detected_sentence.append(predicted_character)
                        sentence_field = ''.join(map(str, detected_sentence))  # Convert to string

                        word_suggestions = [word for word in dictionary if word.startswith(sentence_field.upper())]

                    last_predicted_character = predicted_character
                stable_predictions.clear()

            last_detection_time = time.time()

    display_frame = draw_interface(frame, sentence_field, final_field, word_suggestions)
    cv2.imshow("frame", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()