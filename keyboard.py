# Advanced Hand Gesture Detection for Both Hands with Onscreen Keyboard, Backspace, and Shift using OpenCV and Mediapipe

import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Add Shift key for uppercase/lowercase
keys = [
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', '<-'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
    ['Sft','Z', 'X', 'C', 'V', 'B', 'N', 'M', 'Spc']
]

key_w, key_h = 35, 35  # Smaller key size


def fingers_up(hand_landmarks):
    tips = [
        mp_hands.HandLandmark.THUMB_TIP,
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]
    pips = [
        mp_hands.HandLandmark.THUMB_IP,
        mp_hands.HandLandmark.INDEX_FINGER_PIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
        mp_hands.HandLandmark.RING_FINGER_PIP,
        mp_hands.HandLandmark.PINKY_PIP
    ]
    fingers = []
    for tip, pip in zip(tips, pips):
        if tip == mp_hands.HandLandmark.THUMB_TIP:
            fingers.append(hand_landmarks.landmark[tip].x < hand_landmarks.landmark[pip].x)
        else:
            fingers.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y)
    return fingers  # [thumb, index, middle, ring, pinky]


def draw_keyboard(img, shift_on):
    # Center the keyboard in the middle of the screen
    total_keyboard_height = len(keys) * (key_h + 3)
    total_keyboard_width = max(len(row) for row in keys) * (key_w + 3)
    start_y = (img.shape[0] - total_keyboard_height) // 2
    overlay = img.copy()
    for row_idx, row in enumerate(keys):
        row_width = len(row) * (key_w + 3)
        start_x = (img.shape[1] - row_width) // 2  # Center each row horizontally
        for col_idx, key in enumerate(row):
            x = start_x + col_idx * (key_w + 3)
            y = start_y + row_idx * (key_h + 3)
            # Aquamarine color (BGR)
            color = (212, 255, 127)
            if key == 'Sft' and shift_on:
                color = (255, 255, 127)  # Highlight shift key
            # Draw filled rectangle on overlay
            cv2.rectangle(overlay, (x, y), (x + key_w, y + key_h), color, -1)
            # Draw border (white)
            cv2.rectangle(overlay, (x, y), (x + key_w, y + key_h), (255, 255, 255), 2)
            label = key
            if shift_on and len(key) == 1 and key.isalpha():
                label = key.upper()
            elif not shift_on and len(key) == 1 and key.isalpha():
                label = key.lower()
            font_scale = 0.7 if len(label) == 1 else 0.6
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
            text_x = x + (key_w - text_size[0]) // 2
            text_y = y + (key_h + text_size[1]) // 2
            cv2.putText(overlay, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), 1)
    # Blend overlay with original image for transparency
    alpha = 0.5  # Transparency factor
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img

def get_key_from_point(x, y, img_shape):
    # Center the keyboard in the middle of the screen (match draw_keyboard)
    total_keyboard_height = len(keys) * (key_h + 3)
    for row_idx, row in enumerate(keys):
        row_width = len(row) * (key_w + 3)
        start_x = (img_shape[1] - row_width) // 2  # Center each row horizontally
        start_y = (img_shape[0] - total_keyboard_height) // 2
        key_y = start_y + row_idx * (key_h + 3)
        for col_idx, key in enumerate(row):
            key_x = start_x + col_idx * (key_w + 3)
            if key_x < x < key_x + key_w and key_y < y < key_y + key_h:
                return key
    return None

typed_text = ""
last_key = None  # For debounce
shift_on = False
shift_last = False

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        frame = draw_keyboard(frame, shift_on)

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                fingers = fingers_up(hand_landmarks)
                gesture = "Unknown"

                if fingers == [True, False, False, False, False]:
                    gesture = "Thumbs Up!"
                elif fingers == [False, True, True, False, False]:
                    gesture = "Peace ✌️"
                elif fingers == [False, False, False, False, False]:
                    gesture = "Fist"
                elif fingers == [True, True, True, True, True]:
                    gesture = "Open Palm"
                elif fingers[0] and fingers[1] and not any(fingers[2:]):
                    gesture = "OK"
                elif fingers == [False, False, True, False, False]:
                    gesture = "Middle Finger"
                elif fingers == [False, True, False, False, True]:
                    gesture = "Rock 🤘"
                elif fingers == [True, False, False, False, True]:
                    gesture = "Call Me 🤙"
                elif fingers == [False, True, True, True, True]:
                    gesture = "Four"
                elif fingers == [False, True, True, True, False]:
                    gesture = "Three"
                elif fingers == [False, True, True, False, False]:
                    gesture = "Two"

                # Show gesture label near each hand
                x0 = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * frame.shape[1])
                y0 = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * frame.shape[0])
                cv2.putText(frame, f"Hand {idx+1}: {gesture}", (x0, y0 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                # Onscreen keyboard: If only index finger is up, treat as pointer
                if fingers == [False, True, False, False, False]:
                    ix = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
                    iy = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])
                    cv2.circle(frame, (ix, iy), 10, (255, 0, 255), -1)
                    key = get_key_from_point(ix, iy, frame.shape)
                    if key:
                        cv2.rectangle(frame, (ix-20, iy-20), (ix+20, iy+20), (0,255,255), 1)
                        cv2.putText(frame, f"Selected: {key}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                        # Add key to typed_text if not already last char (debounce)
                        if key != last_key:
                            if key == '<-':
                                typed_text = typed_text[:-1]
                            elif key == 'Spc':
                                typed_text += ' '
                            elif key == 'Sft':
                                shift_on = not shift_on
                            elif len(key) == 1:
                                if shift_on:
                                    typed_text += key.upper()
                                else:
                                    typed_text += key.lower()
                            last_key = key
                    else:
                        last_key = None
                else:
                    last_key = None

        # Show typed text
        cv2.rectangle(frame, (20, 20), (600, 60), (255,255,255), -1)
        cv2.putText(frame, "Typed: " + typed_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)

        # Make window full screen
        cv2.namedWindow("Gesture Detection + Onscreen Keyboard", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Gesture Detection + Onscreen Keyboard", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        cv2.imshow("Gesture Detection + Onscreen Keyboard", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()