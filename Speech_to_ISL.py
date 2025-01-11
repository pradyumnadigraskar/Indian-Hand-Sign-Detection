import cv2
import os
import random
import numpy as np
import speech_recognition as sr
from deep_translator import GoogleTranslator

# Global variable to keep track of scroll position
scroll_position = 0

def load_letter_image(letter):
    if letter == " ":
        return np.ones((128, 128, 3), dtype=np.uint8) * 255  # White image for spaces
    folder_path = f'C:/Users/SANJANA/OneDrive/Desktop/B.TECH_PROJECT/static/data/{letter.upper()}'
    
    # Ensure the folder path is correct
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

def display_images(word):
    word_images = []
    for letter in word:
        image = load_letter_image(letter)
        if image is not None:
            colored_image_area = np.ones((128, 128, 3), dtype=np.uint8) * np.array([255, 228, 196], dtype=np.uint8)
            colored_image_area[0:128, :, :] = image
            labeled_image = np.ones((180, 128, 3), dtype=np.uint8) * np.array([152, 251, 152], dtype=np.uint8)
            labeled_image[0:128, :, :] = colored_image_area
            font_scale = 1.2
            font_thickness = 2
            text_color = (0, 0, 0)
            text_size = cv2.getTextSize(letter.upper(), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            text_x = (labeled_image.shape[1] - text_size[0]) // 2
            text_y = 155
            cv2.putText(labeled_image, letter.upper(), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness, cv2.LINE_AA)
            word_images.append(labeled_image)
    if word_images:
        word_row = np.hstack(word_images)
        return word_row
    else:
        return None

def mouse_scroll(event, x, y, flags, param):
    global scroll_position
    if event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            scroll_position = max(0, scroll_position - 20)  # Scroll up
        else:
            scroll_position += 20  # Scroll down

def display_words_scrollable(words):
    global scroll_position
    rows = []
    max_width = 0
    for word in words:
        word_row = display_images(word)
        if word_row is not None:
            rows.append(word_row)
            max_width = max(max_width, word_row.shape[1])
    for i in range(len(rows)):
        if rows[i].shape[1] < max_width:
            padding = np.ones((rows[i].shape[0], max_width - rows[i].shape[1], 3), dtype=np.uint8) * 255
            rows[i] = np.hstack((rows[i], padding))
    if rows:
        concatenated_image = np.vstack(rows)
        
        # Define viewport size and initial scroll position
        viewport_height = 500  # Size of the scrollable window
        max_scroll_position = max(0, concatenated_image.shape[0] - viewport_height)  # Max position to scroll down
        
        # Set mouse callback for scrolling
        cv2.namedWindow("Scrollable Words")
        cv2.setMouseCallback("Scrollable Words", mouse_scroll)
        
        while True:
            # Ensure scroll position stays within the allowable range
            scroll_position = max(0, min(scroll_position, max_scroll_position))
            
            # Define the viewport based on the current scroll position
            viewport = concatenated_image[scroll_position:scroll_position+viewport_height, :]
            cv2.imshow("Scrollable Words", viewport)

            # Keyboard controls for scrolling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Press 'q' to exit
                break

        cv2.destroyAllWindows()

def speech_to_text():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    print("Please speak a sentence...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)  # Adjusting for ambient noise
        audio = recognizer.listen(source)

    print("Recognizing...")
    try:
        # Attempt to recognize Marathi or Hindi text
        text = recognizer.recognize_google(audio, language="mr-IN")
        print("Detected Marathi Text:", text)
        return text, 'mr'
    except sr.UnknownValueError:
        try:
            # If not Marathi, try Hindi
            text = recognizer.recognize_google(audio, language="hi-IN")
            print("Detected Hindi Text:", text)
            return text, 'hi'
        except sr.UnknownValueError:
            print("Sorry, could not understand the speech.")
            return "", ''
    except sr.RequestError as e:
        print("Could not request results; check your network connection.")
        return "", ''


def translate_to_english(text, lang_code):
    translator = GoogleTranslator(source=lang_code, target='en')
    translated_text = translator.translate(text)
    print("Translated Text:", translated_text)
    return translated_text

def main():
    # Step 1: Take user input (either Marathi or Hindi)
    user_input, language_code = speech_to_text()

    if user_input:
        # Step 2: Translate to English
        english_text = translate_to_english(user_input, language_code)

        # Step 3: Display corresponding ISL images
        words = english_text.split()
        display_words_scrollable(words)

if __name__ == "__main__":
    main()



