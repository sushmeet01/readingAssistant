import cv2
import os
import numpy as np
from datetime import datetime
from inference import InferencePipeline
import multiprocessing
import pyttsx3  
import time
from pitch import AudioPlayer

audio_dir = "audio"
player = AudioPlayer(audio_dir)


pipeline = InferencePipeline()

var = ''


def speak(text):
    engine = pyttsx3.init()
    # Set the speech rate (half the default speed, e.g., 100 if default is 200)
    engine.setProperty('rate', 10)
    # Set the volume to 60%
    engine.setProperty('volume', 0.6)
    # Choose voice 1 (the second available voice)
    voices = engine.getProperty('voices')
    if len(voices) > 1:
        engine.setProperty('voice', voices[1].id)
    else:
        engine.setProperty('voice', voices[0].id)
    engine.say(text)
    engine.runAndWait()


def speak_captions(queue):
    """Consumer process to speak captions from the queue with delays"""
    engine = pyttsx3.init()
    while True:
        caption = queue.get()
        if caption is None:  # Termination signal
            break
        engine.say(caption)
        engine.runAndWait()

def preprocess_image(image, scaling_factor=2.9):
    "Scale, convert to grayscale, and threshold the image for easier contour detection."
    width = int(image.shape[1] * scaling_factor)
    height = int(image.shape[0] * scaling_factor)
    enlarged_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    
    gray = cv2.cvtColor(enlarged_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    white_pixels = np.sum(blurred > 127)
    black_pixels = np.sum(blurred <= 127)
    
    if black_pixels > white_pixels:
        blurred = cv2.bitwise_not(blurred)
    
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return enlarged_image, thresh

def enhance_character_separation(thresh):
    "Use morphological operations to improve character separation in the thresholded image."
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded = cv2.erode(thresh, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    return dilated

def detect_contours(dilated):
    "Detect contours from the dilated image."
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def split_connected_words(x, y, w, h, thresh):
    "Detect and split connected words based on vertical projection gaps."
    roi = thresh[y:y+h, x:x+w]
    v_proj = np.sum(roi, axis=0)
    
    split_positions = []
    current_gap_start = None
    for i in range(len(v_proj)):
        if v_proj[i] == 0:
            if current_gap_start is None:
                current_gap_start = i
        else:
            if current_gap_start is not None:
                if i - current_gap_start > 10:
                    split_positions.append((current_gap_start, i))
                current_gap_start = None

    if not split_positions:
        return [(x, y, w, h)]
    else:
        split_boxes = []
        prev_x = 0
        for start, end in split_positions:
            split_boxes.append((x + prev_x, y, end - prev_x, h))
            prev_x = end
        if prev_x < w:
            split_boxes.append((x + prev_x, y, w - prev_x, h))
        return split_boxes

def filter_and_split_boxes(contours, thresh, min_width=20, max_width=400, min_height=20, max_height=200):
    "Filter contours by size and split boxes as needed."
    word_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        if min_width <= w <= max_width and min_height <= h <= max_height:
            split_boxes = split_connected_words(x, y, w, h, thresh)
            word_boxes.extend(split_boxes)
    return word_boxes

def adjust_large_boxes(word_boxes):
    "Adjust bounding boxes if their heights vary significantly."
    if not word_boxes:
        print("No word boxes found to adjust.")
        return []

    adjusted_boxes = []
    for i in range(len(word_boxes) - 1):
        x1, y1, w1, h1 = word_boxes[i]
        x2, y2, w2, h2 = word_boxes[i + 1]

        if h1 > 2 * h2:
            split_height = h1 // 2
            adjusted_boxes.append((x1, y1, w1, split_height))
            adjusted_boxes.append((x1, y1 + split_height, w1, h1 - split_height))
        else:
            adjusted_boxes.append((x1, y1, w1, h1))
    
    adjusted_boxes.append(word_boxes[-1])
    return adjusted_boxes

def sort_boxes_by_position(word_boxes, line_threshold=20):
    "Sort bounding boxes by vertical and horizontal alignment."
    if not word_boxes:
        print("No boxes to sort.")
        return []

    word_boxes.sort(key=lambda box: (box[1], box[0]))

    sorted_lines = []
    current_line = [word_boxes[0]]
    
    for i in range(1, len(word_boxes)):
        x, y, w, h = word_boxes[i]
        _, prev_y, _, prev_h = word_boxes[i - 1]

        if abs(y - prev_y) <= line_threshold:
            current_line.append(word_boxes[i])
        else:
            sorted_lines.append(sorted(current_line, key=lambda box: box[0]))
            current_line = [word_boxes[i]]
    
    if current_line:
        sorted_lines.append(sorted(current_line, key=lambda box: box[0]))

    sorted_boxes = [box for line in sorted_lines for box in line]
    return sorted_boxes


def save_words_sequentially(image, word_boxes, output_dir, queue, scaling_factor=2.9, invert_for_consistency=True):
    global var
    """Save each word image and put predictions in queue"""
    os.makedirs(output_dir, exist_ok=True)
    existing_files = [f for f in os.listdir(output_dir) if f.startswith('word_') and f.endswith('.png')]
    existing_indices = [int(f.split('_')[1].split('.')[0]) for f in existing_files if f.split('_')[1].split('.')[0].isdigit()]
    current_index = max(existing_indices, default=0)

    for (x, y, w, h) in word_boxes:
        word_image = image[y:y+h, x:x+w]
        original_width = int(w / scaling_factor)
        original_height = int(h / scaling_factor)
        resized_word_image = cv2.resize(word_image, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
        grayscale_word = cv2.cvtColor(resized_word_image, cv2.COLOR_BGR2GRAY)

        if invert_for_consistency and (np.sum(grayscale_word <= 127) > np.sum(grayscale_word > 127)):
            grayscale_word = cv2.bitwise_not(grayscale_word)

        current_index += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(output_dir, f'word_{current_index}_{timestamp}.png')
        cv2.imwrite(filename, grayscale_word)
        
        # Put prediction in queue without blocking
        caption = pipeline.generate_caption(filename, beam_size=3)
        # queue.put(caption)
        var = var + caption + ' '
    print(var)
    player.play_type_2(var, strict=True, overlap=0.65, pitch_shift=2)
    


def main(image_path, queue, output_dir="devanagari_characters", scaling_factor=2.9):
    """Main processing function with queue integration"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    enlarged_image, thresh = preprocess_image(image, scaling_factor)
    dilated = enhance_character_separation(thresh)
    contours = detect_contours(dilated)

    word_boxes = filter_and_split_boxes(contours, thresh)
    if not word_boxes:
        print("No word boxes detected.")
        return

    adjusted_boxes = adjust_large_boxes(word_boxes)
    sorted_word_boxes = sort_boxes_by_position(adjusted_boxes)
    
    if sorted_word_boxes:
        save_words_sequentially(enlarged_image, sorted_word_boxes, output_dir, queue, scaling_factor)
        print(f"Extracted {len(sorted_word_boxes)} words and saved them in '{output_dir}'")

if __name__ == "__main__":
    # Create communication queue and start speech process
    output_queue = multiprocessing.Queue()
    speech_process = multiprocessing.Process(target=speak_captions, args=(output_queue,))
    speech_process.start()

    try:
        IMAGE_PATH = "1.jpg"
        main(IMAGE_PATH, output_queue)
    finally:
        output_queue.put(None)
        speech_process.join()
