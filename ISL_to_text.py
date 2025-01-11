import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

# Load the trained model
model_dict = pickle.load(open(r'C:\Users\rajpu\OneDrive\Desktop\B.TECH_PROJECT\B.TECH_PROJECT_backup_30_12_24\two.p', 'rb'))
model = model_dict['model']
scaler = model_dict['scaler']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)

# Predefined dictionary of words for suggestions
dictionary = ["HELLO", "HELL", "HELMET", "HELP", "HAPPY", "HAND", "PYTHON", "CODE", "SIGN", "LANGUAGE", "AI", "APPLE",
              "BABA", "BAG", "BOMB", "BULLET", "LAPTOP", "MOBILE", "GROUND", "GRAPES", "GOOGLES", "GUITAR", "WATCH",
              "WORK", "TICKET", "BULB", "ALMIRA", "CHAIR", "MONITOR", "COMPUTER", "WINDOW", "DOOR", "PHONE", "TABLE",
              "KEYBOARD", "MOUSE", "PRINTER", "BOOK", "PEN", "NOTE", "PAPER", "FOLDER", "BOTTLE", "WATER", "CUP", "COFFEE",
              "TEA", "PLATE", "SPOON", "FORK", "KNIFE", "FOOD", "CHAIR", "BED", "CUSHION", "PILLOW", "MIRROR", "LIGHT",
              "FAN", "AC", "ROOM", "HOUSE", "DOCTOR", "NURSE", "TEACHER", "STUDENT", "CLASS", "BOARD", "MARKER",
              "ERASER", "SCHOOL", "COLLEGE", "UNIVERSITY", "HOSPITAL", "CLINIC", "MEDICINE", "SURGERY", "INJECTION",
              "VACCINE", "IM", "SWARAJ", "PARTH", "HEALTH", "BODY", "HEAD", "NECK", "SHOULDER", "ARM", "HAND", "FINGER",
              "LEG", "FOOT", "HEART", "BRAIN", "LUNG", "PRADYUMNA", "LIVER", "STOMACH", "KIDNEY", "BONE", "MUSCLE",
              "JOINT", "TOOTH", "EYE", "EAR", "NOSE", "MOUTH", "SKIN", "HAIR", "FEVER", "COUGH", "COLD", "PAIN", "FRACTURE",
              "BLOOD", "URINE", "XRAY", "MRI", "SCAN", "SURGERY", "MEDICINE", "PRESCRIPTION", "DOCTOR", "PATIENT", "BED",
              "WARD", "AMBULANCE", "RESCUE", "POLICE", "FIRE", "ENGINE", "VEHICLE", "CAR", "BIKE", "TRUCK", "BUS", "TRAIN",
              "AIRPLANE", "SHIP", "BOAT", "CYCLE", "WHEEL", "TYRE", "ENGINE", "BRAKE", "SPEED", "GEAR", "FUEL", "OIL",
              "PETROL", "DIESEL", "BATTERY", "ELECTRIC", "POWER", "CHARGE", "LIGHT", "SWITCH", "SOCKET", "CABLE", "WIRE",
              "PLUG", "FUSE", "CURRENT", "VOLTAGE", "AMPERE", "WATT", "ENERGY", "HEAT", "COOL", "WATER", "AIR", "FIRE",
              "EARTH", "SKY", "SUN", "MOON", "STAR", "PLANET", "GALAXY", "UNIVERSE", "ATOM", "MOLECULE", "ELEMENT", "COMPOUND",
              "SUBSTANCE", "MATERIAL", "SOLID", "LIQUID", "GAS", "PLASMA", "CHEMISTRY", "PHYSICS", "BIOLOGY", "SCIENCE", 
              "MATHEMATICS", "GEOMETRY", "ALGEBRA", "CALCULUS", "TRIGONOMETRY", "PROBABILITY", "STATISTICS", "COMPUTER", 
              "PROGRAM", "SOFTWARE", "HARDWARE", "DATABASE", "SERVER", "NETWORK", "SECURITY", "ENCRYPTION", "DECRYPTION", 
              "ALGORITHM", "MODEL", "DATA", "ARTIFICIAL", "INTELLIGENCE", "LEARNING", "TRAINING", "TESTING", "VALIDATION", 
              "ACCURACY", "LOSS", "OPTIMIZATION", "PARAMETER", "FEATURE", "INPUT", "OUTPUT", "LAYER", "NEURON", "CONNECTION", 
              "WEIGHT", "BIAS", "ACTIVATION", "FUNCTION", "LINEAR", "NONLINEAR", "RELATION", "SYSTEM", "PROCESS", "OPERATION", 
              "CONTROL", "MANAGEMENT", "PLANNING", "DESIGN", "DEVELOPMENT", "TESTING", "DEPLOYMENT", "MAINTENANCE", "UPGRADE", 
              "VERSION", "RELEASE", "DOCUMENT", "REPORT", "PRESENTATION", "MEETING", "DISCUSSION", "DECISION", "ACTION", 
              "TASK", "PROJECT", "TEAM", "MEMBER", "ROLE", "RESPONSIBILITY", "GOAL", "OBJECTIVE", "STRATEGY", "POLICY", 
              "RULE", "REGULATION", "LAW", "ORDER", "COMMAND", "EXECUTION", "PERFORMANCE", "RESULT", "OUTPUT", "FEEDBACK", 
              "IMPROVEMENT", "SUCCESS", "FAILURE", "PROGRESS", "ACHIEVEMENT", "MILESTONE", "DEADLINE", "TIME", "DATE", "DAY", 
              "WEEK", "MONTH", "YEAR", "HOUR", "MINUTE", "SECOND", "CLOCK", "CALENDAR", "EVENT", "SCHEDULE", "PLAN", "ACTIVITY", 
              "WORK", "BREAK", "REST", "SLEEP", "WAKE", "MORNING", "AFTERNOON", "EVENING", "NIGHT", "BREAKFAST", "LUNCH", "DINNER",
              "MEAL", "FOOD", "DRINK", "WATER", "JUICE", "TEA", "COFFEE", "MILK", "SUGAR", "SALT", "SPICE", "VEGETABLE", "FRUIT",
              "GRAIN", "MEAT", "FISH", "EGG", "DAIRY", "BUTTER", "CHEESE", "BREAD", "CAKE", "PASTA", "RICE", "SOUP", "SALAD",
              "SNACK", "DESSERT", "CHOCOLATE", "CANDY", "ICE", "CREAM", "COOKIE", "BISCUIT", "CRACKER", "CHIP", "POP", "CORN", 
              "NUT", "SEED", "BEAN", "PULSE", "CROP", "FIELD", "FARM", "ANIMAL", "BIRD", "FISH", "INSECT", "PLANT", "TREE", "FLOWER",
              "LEAF", "ROOT", "STEM", "SEED", "SOIL", "EARTH", "WATER", "AIR", "FIRE", "LIGHT", "ENERGY", "POWER", "ELECTRICITY", 
              "CURRENT", "VOLTAGE", "MAGNET", "FORCE", "MOTION", "SPEED", "ACCELERATION"]

# Sentence tracking
detected_sentence = []  # List to store detected characters
sentence_field = ""  # Variable to hold the sentence as a string
final_field = ""  # Variable to hold the final accepted sentence
word_suggestions = []  # List to hold word suggestions

last_predicted_character = None
stable_predictions = []
cooldown_time = 0.3
last_detection_time = time.time()

# Button positions for suggestions, clear, and space buttons
button_positions = []
clear_button_position = None
space_button_position = None


def draw_interface(frame, sentence, final_sentence, suggestions):
    """
    Draw the interface with the camera feed, sentence, final sentence, and suggestions.
    """
    global button_positions, clear_button_position, space_button_position
    button_positions.clear()

    # Create a larger frame to hold the camera feed and suggestion panel
    panel_height = 200
    display_frame = np.zeros((frame.shape[0] + panel_height, frame.shape[1], 3), dtype=np.uint8)

    # Copy the camera feed to the top portion
    display_frame[:frame.shape[0], :] = frame

    # Draw the sentence area
    cv2.rectangle(display_frame, (0, frame.shape[0]), (frame.shape[1], frame.shape[0] + panel_height), (255, 255, 255), -1)
    cv2.putText(display_frame, "Detected Sentence: " + ''.join(sentence),
                (10, frame.shape[0] + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.putText(display_frame, "Final Sentence: " + final_sentence,
                (10, frame.shape[0] + 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.putText(display_frame, "Suggestions: ",
                (10, frame.shape[0] + 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    for i, suggestion in enumerate(suggestions[:5]):
        cv2.putText(display_frame, f"{i + 1}. {suggestion}",
                    (10, frame.shape[0] + 150 + (i * 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Define clear and space buttons (as colored rectangles)
    button_width, button_height = 200, 40
    clear_button_position = (frame.shape[1] // 2 - button_width // 2, frame.shape[0] + 40)
    space_button_position = (frame.shape[1] // 2 - button_width // 2, frame.shape[0] + 90)

    # Draw buttons
    cv2.rectangle(display_frame, clear_button_position, (clear_button_position[0] + button_width, clear_button_position[1] + button_height), (0, 255, 0), -1)
    cv2.putText(display_frame, "Clear", (clear_button_position[0] + 50, clear_button_position[1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.rectangle(display_frame, space_button_position, (space_button_position[0] + button_width, space_button_position[1] + button_height), (255, 0, 0), -1)
    cv2.putText(display_frame, "Space", (space_button_position[0] + 50, space_button_position[1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return display_frame


def mouse_callback(event, x, y, flags, param):
    """
    Handle mouse events for clearing or adding space in the detected sentence.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        if clear_button_position[0] <= x <= clear_button_position[0] + 200 and clear_button_position[1] <= y <= clear_button_position[1] + 40:
            detected_sentence.clear()
            word_suggestions.clear()
        elif space_button_position[0] <= x <= space_button_position[0] + 200 and space_button_position[1] <= y <= space_button_position[1] + 40:
            detected_sentence.append(" ")


# Set up OpenCV mouse callback
cv2.setMouseCallback('Hand Gesture Recognition', mouse_callback)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame to create mirror image
    frame = cv2.flip(frame, 1)

    # Get hand landmarks from the frame
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                                      mp_drawing_styles.get_default_hand_landmarks_style(), 
                                      mp_drawing_styles.get_default_hand_connections_style())

            # Process hand landmarks and predict gesture
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])

            # Make gesture prediction
            input_data = np.array(landmarks).flatten().reshape(1, -1)
            input_data = scaler.transform(input_data)  # Scale the input
            prediction = model.predict(input_data)

            # Add predicted gesture to stable_predictions buffer
            stable_predictions.append(prediction[0])

            # If we have enough predictions, get the most frequent one
            if len(stable_predictions) > 5:
                stable_predictions.pop(0)

            # Get the most frequent prediction (majority vote)
            most_common_prediction = max(set(stable_predictions), key=stable_predictions.count)
            
            if time.time() - last_detection_time > cooldown_time:
                if most_common_prediction != last_predicted_character:
                    detected_sentence.append(most_common_prediction)
                    word_suggestions = [word for word in dictionary if word.startswith("".join(detected_sentence))]
                last_predicted_character = most_common_prediction
                last_detection_time = time.time()

    # Prepare the display frame
    final_sentence = "".join(detected_sentence)  # Join the sentence list
    display_frame = draw_interface(frame, detected_sentence, final_sentence, word_suggestions)

    cv2.imshow('Hand Gesture Recognition', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
