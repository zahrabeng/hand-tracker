import cv2
import mediapipe as mp
import numpy as np
import time


class HandDetector():
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.finger_tip_ids = [4, 8, 12, 16, 20]  # Tip landmark IDs for thumb to pinky

    def find_hands(self, image, draw=True):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(image_rgb)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return image

    def find_hand_landmarks(self, image, hand_index=0, draw=True):
        self.landmark_list = []
        if self.results.multi_hand_landmarks:
            selected_hand = self.results.multi_hand_landmarks[hand_index]
            for id, landmark in enumerate(selected_hand.landmark):
                height, width, _ = image.shape
                pixel_x, pixel_y = int(landmark.x * width), int(landmark.y * height)
                self.landmark_list.append([id, pixel_x, pixel_y])
                if draw:
                    cv2.circle(image, (pixel_x, pixel_y), 15, (0, 255, 236), cv2.FILLED)
        return self.landmark_list

    def get_fingers_up(self):
        fingers = []

        # Thumb (check horizontal direction)
        if self.landmark_list[self.finger_tip_ids[0]][1] < self.landmark_list[self.finger_tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other 4 fingers (check vertical direction)
        for i in range(1, 5):
            if self.landmark_list[self.finger_tip_ids[i]][2] < self.landmark_list[self.finger_tip_ids[i] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

