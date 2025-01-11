
from flask import Flask, render_template, Response, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import pickle
from gtts import gTTS
import pyttsx3
import time
import random
import os
import speech_recognition as sr
from deep_translator import GoogleTranslator
import base64


app = Flask(__name__)

ISL_IMAGES_FOLDER = r'C:\Users\rajpu\OneDrive\Desktop\B.TECH_PROJECT\B.TECH_PROJECT_backup_30_12_24\static\data'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/options')
def options():
    return render_template('options.html')


#first module routes 
#Load the trained model
model_dict = pickle.load(open('two.p', 'rb'))
model = model_dict['model']
scaler = model_dict['scaler']

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7)

engine = pyttsx3.init()

# Variables
dictionary = ["HELLO", "HELL", "HELMET", "HELP", "HAPPY", "HAND", "PYTHON"]
detected_sentence = []
sentence_field = ""
final_field = ""
word_suggestions = []
last_predicted_character = None
stable_predictions = []
cooldown_time = 0.2
last_detection_time = 0

cap = cv2.VideoCapture(0)

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/Index')
def Index():
    return render_template('home.html')

@app.route('/options')
def options1():
    return render_template('options.html')

@app.route('/about')
def about():
    return render_template('about_us.html')

@app.route('/contact')
def contact():
    return render_template('contact_us.html')

@app.route('/module1')
def module1():
    return render_template('module1.html')

def gen_frames():
    global detected_sentence, sentence_field, final_field, word_suggestions, last_predicted_character, last_detection_time
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Draw hand landmarks on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Process landmarks and make predictions
                data_aux = []
                x_, y_ = [], []
                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))

                while len(data_aux) < 126:
                    data_aux.extend([0, 0])

                data_input = np.asarray(data_aux).reshape(1, -1)
                data_input_scaled = scaler.transform(data_input)
                prediction = model.predict(data_input_scaled)
                predicted_character = prediction[0]

                # Handle stable predictions
                if time.time() - last_detection_time >= cooldown_time:
                    stable_predictions.append(predicted_character)

                    if len(stable_predictions) >= 3:
                        if stable_predictions[0] == stable_predictions[1] == stable_predictions[2]:
                            if predicted_character != last_predicted_character:
                                detected_sentence.append(predicted_character)
                                sentence_field = ''.join(map(str, detected_sentence))
                                final_field = sentence_field  # Update final sentence
                                word_suggestions = [word for word in dictionary if word.startswith(sentence_field.upper())]
                                last_predicted_character = predicted_character
                        stable_predictions.clear()

                    last_detection_time = time.time()

                # Overlay predicted character on the frame
                cv2.putText(frame, f"Predicted: {predicted_character}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode frame for display in the web browser
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/clear', methods=['POST'])
def clear():
    global detected_sentence, sentence_field, final_field
    detected_sentence.clear()
    sentence_field = ""
    final_field = ""
    return jsonify({"status": "cleared"})

@app.route('/speak', methods=['POST'])
def speak():
    global final_field
    text = final_field
    if text:
        engine.say(text)
        engine.runAndWait()
    return jsonify({"status": "spoken"})

@app.route('/get_sentence')
def get_sentence():
    global sentence_field, final_field
    return jsonify({"sentence": sentence_field, "final_sentence": final_field})






#second module routes

@app.route('/module2')
def module2():
    return render_template('module2.html')

def load_letter_image(letter):
    if letter == " ":
        return np.ones((128, 128, 3), dtype=np.uint8) * 255  # White image for spaces
    
    folder_path = os.path.join(ISL_IMAGES_FOLDER, letter.upper())
    if not os.path.exists(folder_path):
        print(f"Folder not found for letter: {letter.upper()} at path: {folder_path}")
        return None

    images = os.listdir(folder_path)
    if images:
        random_image = random.choice(images)
        image_path = os.path.join(folder_path, random_image)
        image = cv2.imread(image_path)
        return image
    else:
        print(f"No images found in folder: {folder_path}")
        return None

@app.route('/start_recording', methods=['POST'])
def start_recording():
    try:
        user_input, language_code = speech_to_text()
        if not user_input:
            return jsonify({'message': 'Recording complete, but no speech detected.', 'transcribed_text': '', 'image_data': []})

        # Translate to English
        english_text = translate_to_english(user_input, language_code)
        if english_text == "Translation Error":
            return jsonify({'message': 'Translation error occurred.', 'transcribed_text': '', 'image_data': []})

        # Generate ISL images
        words = english_text.split()
        word_images = []
        for word in words:
            for letter in word:
                img = load_letter_image(letter)
                if img is not None:
                    word_images.append(img)

        # Convert images to base64
        image_data = []
        for img in word_images:
            if img is not None:
                _, buffer = cv2.imencode('.jpg', img)
                img_str = base64.b64encode(buffer).decode('utf-8')
                image_data.append(img_str)

        return jsonify({
            'message': 'Recording and transcription complete!',
            'transcribed_text': english_text,
            'image_data': image_data
        })
    except Exception as e:
        print(f"Error in start_recording: {e}")
        return jsonify({'message': 'An error occurred.', 'transcribed_text': '', 'image_data': []})

def speech_to_text():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    print("Please speak a sentence...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=10)  # Wait for up to 10 seconds
        except Exception as e:
            print(f"Error during audio capture: {e}")
            return "", ""

    print("Recognizing...")
    try:
        text = recognizer.recognize_google(audio, language="mr-IN")
        print("Detected Marathi Text:", text)
        return text, 'mr'
    except sr.UnknownValueError:
        try:
            text = recognizer.recognize_google(audio, language="hi-IN")
            print("Detected Hindi Text:", text)
            return text, 'hi'
        except sr.UnknownValueError:
            print("Sorry, could not understand the speech.")
            return "", ""
    except sr.RequestError as e:
        print(f"RequestError in speech recognition: {e}")
        return "", ""

def translate_to_english(text, lang_code):
    try:
        translator = GoogleTranslator(source=lang_code, target='en')
        translated_text = translator.translate(text)
        print("Translated Text:", translated_text)
        return translated_text
    except Exception as e:
        print(f"Error in translation: {e}")
        return "Translation Error"


if __name__ == '__main__':
    app.run(debug=True)
