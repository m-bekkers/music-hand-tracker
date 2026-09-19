import argparse
import urllib.request
from pathlib import Path
from pprint import pprint
import math

import cv2
from cv2.typing import *
import mediapipe as mp
import numpy as np

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

def determine_chord(hand_landmarks, hand_world_landmarks, frame):#, handedness, hand_connections):
    #pprint(result)
    #pass
    # Right hand controls the chord being played

    def s(a, b): return math.sqrt(pow((a.x - b.x), 2) + pow((a.y - b.y), 2))

    height, width = frame.shape[:2]

    I_EXTENDED = 0.045
    M_EXTENDED = 0.045
    R_EXTENDED = 0.045
    P_EXTENDED = 0.037
    T_EXTENDED = 0.009
    M_X_EXTENDED = 0.0085
    M_Y_EXTENDED = 0.0085

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

    center_x = int((WRIST.x + T_B.x + I_B.x + M_B.x + R_B.x + P_B.x) / 6 * width)
    center_y = int((WRIST.y + T_B.y + I_B.y + M_B.y + R_B.y + P_B.y) / 6 * height)

    # Draw a circle there so it is definitely visible in the current frame.
    cv2.circle(frame, (center_x, center_y), 18, (255, 0, 0), 2)
    cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

    # Calculate distances from the palm to fingertips
    d_I_to_C = s(I_T, I_B)

    return "No chord"

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  hand_world_landmarks_list = detection_result.hand_world_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

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

    #print(mp_hands.HAND_CONNECTIONS)

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    hand_text = determine_chord(hand_landmarks, hand_world_landmarks, annotated_image)

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, hand_text,#f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image


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

    timestamp_ms = 0
    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
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
                annotated_image = draw_landmarks_on_image(frame, result)
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