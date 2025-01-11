from flask import Flask, render_template, Response, request, jsonify
import cv2
import pickle
import numpy as np
import mediapipe as mp
import pyttsx3

# Initialize Flask app
app = Flask(__name__)

# Initialize pyttsx3 for text-to-speech
engine = pyttsx3.init()

# Load trained model
model_dict = pickle.load(open('three.p', 'rb'))
model = model_dict['model']
scaler = model_dict['scaler']

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7)

# Sentence tracking
sentence_field = ""
final_field = ""
word_suggestions = []
detected_sentence = []

# Predefined dictionary
dictionary = ["HELLO", "HELL", "HELMET", "HELP", "HAPPY", "HAND", "PYTHON"]

# Initialize video capture
cap = cv2.VideoCapture(0)

# Function to generate video feed
def generate_frames():
    global sentence_field, final_field, word_suggestions, detected_sentence

    while True:
        success, frame = cap.read()
        if not success:
            break

        data_aux = []
        x_ = []
        y_ = []

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
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

            detected_sentence.append(predicted_character)
            sentence_field = ''.join(map(str, detected_sentence))
            word_suggestions = [word for word in dictionary if word.startswith(sentence_field.upper())]

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/button_action', methods=['POST'])
def button_action():
    global sentence_field, final_field, detected_sentence
    action = request.json.get('action')

    if action == 'clear':
        detected_sentence.clear()
        sentence_field = ""
        final_field = ""
    elif action == 'space':
        final_field += sentence_field + " "
        detected_sentence.clear()
        sentence_field = ""
    elif action == 'speak':
        if final_field:
            engine.say(final_field)
            engine.runAndWait()

    return jsonify({'sentence': sentence_field, 'final': final_field})

if __name__ == '__main__':
    app.run(debug=True)
