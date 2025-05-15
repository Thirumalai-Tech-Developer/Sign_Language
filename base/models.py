from django.db import models
import mediapipe as mp
import cv2
import csv

mp_hand = mp.solutions.hands
hands = mp_hand.Hands(static_image_mode=False, max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

def GraphFramemrk(video_path, out_csv):
    
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    with open(out_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        header = ['frame']
        for hand_idx in range(2):
            for point in range(21):
                header += [f'hand{hand_idx}_x{point}', f'hand{hand_idx}_y{point}', f'hand{hand_idx}_z{point}']
        writer.writerow(header)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            row = [frame_idx]
            hand_landmarks = results.multi_hand_landmarks or []
            for h in range(2):
                if h < len(hand_landmarks):
                    for lm in hand_landmarks[h].landmark:
                        row += [lm.x, lm.y, lm.z]
                    # Removed drawing on frame here
                    # mp.draw_landmarks(frame, hand_landmarks[h], mp_hand.HAND_CONNECTIONS)
                else:
                    row += [None] * 63

            writer.writerow(row)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
