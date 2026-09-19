import argparse
import urllib.request
from pathlib import Path
import math
import time as t

import cv2
from cv2.typing import *
import mediapipe as mp
import numpy as np

import chord_builder

'''

Beige = Thumb
Purple = Index
Orange = Middle
Green = Ring
Blue = Pinky

'''

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path.home() / ".cache" / "music-hand-tracker" / "hand_landmarker.task"
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

mp_drawing = mp.tasks.vision.drawing_utils
mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing_styles = mp.tasks.vision.drawing_styles

timer_start = 0
timer_curr = 0

# Download the MediaPipe hand landmarker model if it is not cached locally.
def ensure_model():
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading hand landmarker model to {MODEL_PATH}")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


# Draw hand landmarks and connections on the frame.
def draw_landmarks(frame, hand_landmarks):
    height, width = frame.shape[:2]
    points = [
        (int(landmark.x * width), int(landmark.y * height))
        for landmark in hand_landmarks
    ]

    connections = (
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17),
    )
    for start, end in connections:
        cv2.line(frame, points[start], points[end], (0, 255, 0), 2)
    for point in points:
        cv2.circle(frame, point, 4, (0, 0, 255), -1)

def what_chord(one, two, three, four, five, six, seven):
    '''
    Based on fingertip distance flags, return the chord held up.
    '''

    if seven:
        return "Seven"
    if six:
        return "Six"

    if (two or three or four or five) and not (one) \
        or \
       (three or four or five) and not (one and two) \
        or \
       (four or five) and not (one and two and three) \
        or \
       (five) and not (one and two and three and four):
        return "No Chord"
    
    if five:
        return "Five"
    if four:
        return "Four"
    if three:
        return "Three"
    if two:
        return "Two"
    if one:
        return "One"

    return "No Chord"

def determine_chord(hand_landmarks, hand_world_landmarks, frame) -> tuple[str, int]:#, handedness, hand_connections):
    # Right hand controls the chord being played

    def s_c(a_x, a_y, b_x, b_y): return math.sqrt(pow((a_x - b_x), 2) + pow((a_y - b_y), 2))

    height, width = frame.shape[:2]

    one, two, three, four, five, six, seven = False, False, False, False, False, False, False

    I_EXTENDED = 85
    M_EXTENDED = 85
    R_EXTENDED = 85
    P_EXTENDED = 85
    T_EXTENDED = 85
    I_T_TOGETHER = 20
    M_T_TOGETHER = 25

    # Use image-space landmarks for the rendered circle so it appears at the palm center on screen.
    WRIST = hand_landmarks[0]
    T_B = hand_landmarks[1]
    T_T = hand_landmarks[4]
    I_B = hand_landmarks[5]
    I_T = hand_landmarks[8]
    M_B = hand_landmarks[9]
    M_T = hand_landmarks[12]
    R_B = hand_landmarks[13]
    R_T = hand_landmarks[16]
    P_B = hand_landmarks[17]
    P_T = hand_landmarks[20]

    # Determine center of palm
    c_x = int((WRIST.x + T_B.x + I_B.x + M_B.x + R_B.x + P_B.x) / 6 * width)
    c_y = int((WRIST.y + T_B.y + I_B.y + M_B.y + R_B.y + P_B.y) / 6 * height)
    c_z = int((WRIST.z + T_B.z + I_B.z + M_B.z + R_B.z + P_B.z) / 6 * width)

    # Helper function to convert all points to a common format
    def convert_to_x_y(point): return int(point.x * width), int(point.y * height)

    I_T_x, I_T_y = convert_to_x_y(I_T)
    M_T_x, M_T_y = convert_to_x_y(M_T)
    R_T_x, R_T_y = convert_to_x_y(R_T)
    P_T_x, P_T_y = convert_to_x_y(P_T)
    T_T_x, T_T_y = convert_to_x_y(T_T)

    # Draw a circle there so it is definitely visible in the current frame.
    cv2.circle(frame, (c_x, c_y), 18, (255, 0, 0), 2)
    cv2.circle(frame, (c_x, c_y), 5, (255, 0, 0), -1)

    # Calculate distances from the palm to fingertips
    d_I_to_C = s_c(I_T_x, I_T_y, c_x, c_y)
    d_M_to_C = s_c(M_T_x, M_T_y, c_x, c_y)
    d_R_to_C = s_c(R_T_x, R_T_y, c_x, c_y)
    d_P_to_C = s_c(P_T_x, P_T_y, c_x, c_y)
    d_T_to_C = s_c(T_T_x, T_T_y, c_x, c_y)

    # For six and seven, we need the distance from the fingertip of the index and middle to the fingertip of the thumb
    d_I_T_to_T_T = s_c(I_T_x, I_T_y, T_T_x, T_T_y)
    d_M_T_to_T_T = s_c(M_T_x, M_T_y, T_T_x, T_T_y)

    #cv2.line(frame, (int(I_T.x * width), int(I_T.y * height)), (c_x, c_y), (0, 255, 0), 2)

    # Check distances from fingertips to center of palm
    if d_I_to_C > I_EXTENDED:
        one = True
    if d_M_to_C > M_EXTENDED:
        two = True
    if d_R_to_C > R_EXTENDED:
        three = True
    if d_P_to_C > P_EXTENDED:
        four = True
    if d_T_to_C > T_EXTENDED:
        five = True
    if d_I_T_to_T_T < I_T_TOGETHER:
        six = True
    if d_M_T_to_T_T < M_T_TOGETHER:
        seven = True

    chord = what_chord(one, two, three, four, five, six, seven)

    match chord:
        case "One":
            chord_num = 1
        case "Two":
            chord_num = 2
        case "Three":
            chord_num = 3
        case "Four":
            chord_num = 4
        case "Five":
            chord_num = 5
        case "Six":
            chord_num = 6
        case "Seven":
            chord_num = 7
        case _:
            chord_num = -1

    return chord, chord_num

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    hand_world_landmarks_list = detection_result.hand_world_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    chord = -1

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        hand_world_landmarks = hand_world_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        hand_text, chord = determine_chord(hand_landmarks, hand_world_landmarks, annotated_image)

        cv2.putText(annotated_image, hand_text,#f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image, chord

def process_chord(chord, last_chord):

    # We need to add a buffer to chord changes to prevent model errors from changing the chord strangely

    if chord != -1 and chord != last_chord:
        global timer_start
        global timer_curr

        if timer_start == 0:
            timer_start = int(t.time() * 1000)
        timer_curr = int(t.time() * 1000)

        if (timer_curr - timer_start < 100):

            timer_curr = int(t.time() * 1000)
            print(f"timer_start: {timer_start} | timer_curr: {timer_curr}, | diff: {timer_curr - timer_start}")

        else:

            chord_name = chord_builder.get_chord_note_names(chord)
            chord_buffer = chord_builder.prepare_chord_buffer(chord_name)
            chord_builder.set_audio_buffer(chord_buffer)
            last_chord = chord
            timer_start = 0

# Initialize MediaPipe
def main(url):
    ensure_model()
    base_options = mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
    )

    # WSL is annoying and does not have direct access to my camera. Start a proxy server...
    vidcap = cv2.VideoCapture(url)

    # Error handling
    if not vidcap.isOpened():
        raise RuntimeError(f"Failed to connect to stream at {url}")

    stream = chord_builder.start_stream()
    timestamp_ms = 0
    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:

            last_chord = -1

            while True:
                ret, frame = vidcap.read()

                # Error handling
                if not ret:
                    break

                # Convert image colour to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                # Process frame for hand tracking
                result = landmarker.detect_for_video(image, timestamp_ms)
                timestamp_ms += 1

                # Draw the landmarks on the frame
                annotated_image, chord = draw_landmarks_on_image(rgb_frame, result)
                annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                #for hand_landmarks in result.hand_landmarks:
                #    draw_landmarks(frame, hand_landmarks)

                cv2.putText(
                    frame,
                    "Hello World",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                # Resize frame to desired size
                resized_frame = cv2.resize(frame, (960, 540))

                # Display resized frame
                cv2.imshow("Hand Tracking", annotated_image)

                process_chord(chord, last_chord)

                # Pressing 'q' exits the loop
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        vidcap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://192.168.0.199:8080/video",
                        help="URL of the webcam stream server")
    main(parser.parse_args().url)