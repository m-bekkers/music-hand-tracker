import argparse
import urllib.request
from pathlib import Path
from pprint import pprint

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

def determine_chord(hand_world_landmarks):#, handedness, hand_connections):
    #pprint(result)
    #pass
    # Right hand controls the chord being played


    I_EXTENDED = 0.045
    M_EXTENDED = 0.045
    R_EXTENDED = 0.045
    P_EXTENDED = 0.037
    T_EXTENDED = 0.009
    M_X_EXTENDED = 0.0085
    M_Y_EXTENDED = 0.0085

    t_x = hand_world_landmarks[4].x
    t_y = hand_world_landmarks[4].y
    t_m_x = hand_world_landmarks[3].x
    t_m_y = hand_world_landmarks[3].y
    t_b = hand_world_landmarks[2].y

    i_x = hand_world_landmarks[8].x
    i_y = hand_world_landmarks[8].y
    i_m_x = hand_world_landmarks[6].x
    i_m_y = hand_world_landmarks[6].y
    i_b = hand_world_landmarks[5].y

    m_x = hand_world_landmarks[12].x
    m_y = hand_world_landmarks[12].y
    m_m_x = hand_world_landmarks[10].x
    m_m_y = hand_world_landmarks[10].y
    m_b = hand_world_landmarks[9].y
    r_x = hand_world_landmarks[16].x
    r_y = hand_world_landmarks[16].y
    r_m_x = hand_world_landmarks[14].x
    r_m_y = hand_world_landmarks[14].y
    r_b = hand_world_landmarks[13].y
    p_x = hand_world_landmarks[20].x
    p_y = hand_world_landmarks[20].y
    p_m_x = hand_world_landmarks[18].x
    p_m_y = hand_world_landmarks[18].y
    p_b = hand_world_landmarks[17].y

    # If landmark 8 (tip of index) is sufficiently far from the palm, we display text saying so
    #print(hand_world_landmarks[8].y)
    i = (
        (abs(i_b - i_x) > I_EXTENDED or abs(i_b - i_y) > I_EXTENDED)
        and (abs(i_b - i_m_x) > M_X_EXTENDED
             or abs(i_b - i_m_y) > M_Y_EXTENDED)
    )
    m = (
        (abs(m_b - m_x) > M_EXTENDED or abs(m_b - m_y) > M_EXTENDED)
        and (abs(m_b - m_m_x) > M_X_EXTENDED
             or abs(m_b - m_m_y) > M_Y_EXTENDED)
    )
    r = (
        (abs(r_b - r_x) > R_EXTENDED or abs(r_b - r_y) > R_EXTENDED)
        and (abs(r_b - r_m_x) > M_X_EXTENDED
             or abs(r_b - r_m_y) > M_Y_EXTENDED)
    )
    p = (
        (abs(p_b - p_x) > P_EXTENDED or abs(p_b - p_y) > P_EXTENDED)
        and (abs(p_b - p_m_x) > M_X_EXTENDED
             or abs(p_b - p_m_y) > M_Y_EXTENDED)
    )
    t = (
        (abs(t_b - t_x) > T_EXTENDED or abs(t_b - t_y) > T_EXTENDED)
        and (abs(t_b - t_m_x) > M_X_EXTENDED
             or abs(t_b - t_m_y) > M_Y_EXTENDED)
    )

    one = i
    two = one and m
    three = two and r
    four = three and p
    five = four and t

    #one = ((abs(i_b - i_x) > EXTENDED) and (abs(i_b - i_m_x) > M_X_EXTENDED)) or ((abs(i_b - i_y) > EXTENDED) and abs(i_b - i_m_y > M_Y_EXTENDED))

    print(
        "\n"
        "Finger diagnostics:\n"
        "Finger   Base Y       Tip X        Tip Y        Tip dX       Tip dY       Middle dX    Middle dY    Extended\n"
        f"Thumb   {t_b: .8f}  {t_x: .8f}  {t_y: .8f}  {abs(t_b - t_x): .8f}  {abs(t_b - t_y): .8f}  {abs(t_b - t_m_x): .8f}  {abs(t_b - t_m_y): .8f}  {str(t):>8}\n"
        f"Index   {i_b: .8f}  {i_x: .8f}  {i_y: .8f}  {abs(i_b - i_x): .8f}  {abs(i_b - i_y): .8f}  {abs(i_b - i_m_x): .8f}  {abs(i_b - i_m_y): .8f}  {str(i):>8}\n"
        f"Middle  {m_b: .8f}  {m_x: .8f}  {m_y: .8f}  {abs(m_b - m_x): .8f}  {abs(m_b - m_y): .8f}  {abs(m_b - m_m_x): .8f}  {abs(m_b - m_m_y): .8f}  {str(m):>8}\n"
        f"Ring    {r_b: .8f}  {r_x: .8f}  {r_y: .8f}  {abs(r_b - r_x): .8f}  {abs(r_b - r_y): .8f}  {abs(r_b - r_m_x): .8f}  {abs(r_b - r_m_y): .8f}  {str(r):>8}\n"
        f"Pinky   {p_b: .8f}  {p_x: .8f}  {p_y: .8f}  {abs(p_b - p_x): .8f}  {abs(p_b - p_y): .8f}  {abs(p_b - p_m_x): .8f}  {abs(p_b - p_m_y): .8f}  {str(p):>8}\n"
        f"\nChords: one={one}  two={two}  three={three}  four={four} five={five}\n"
    )

    if five:
        return "five"
    if four:
        return "four"
    if three:
        return "three"
    if two:
        return "two"
    if one:
        return "one"

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

    hand_text = determine_chord(hand_world_landmarks)

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