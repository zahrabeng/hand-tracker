import cv2
import numpy as np
import HandTrackingModule as htm


def main():
    cap = cv2.VideoCapture(0)
    hand_detector = htm.HandDetector(detection_confidence=0.85)

    previous_x, previous_y = 0, 0
    brush_thickness = 15
    canvas = np.zeros((400, 680, 3), np.uint8)

    while True:
        success, frame = cap.read()
        frame = hand_detector.find_hands(frame)
        landmark_list = hand_detector.find_hand_landmarks(frame, draw=False)

        if len(landmark_list) != 0:
            # Index finger tip
            index_finger_x, index_finger_y = landmark_list[8][1:]

            fingers = hand_detector.get_fingers_up()

            # Drawing mode: Only index finger up
            if fingers[1] and not fingers[2]:
                cv2.circle(frame, (index_finger_x, index_finger_y), 15, (233, 37, 0), cv2.FILLED)
                if previous_x == 0 and previous_y == 0:
                    previous_x, previous_y = index_finger_x, index_finger_y

                # Draw on both image and canvas
                cv2.line(frame, (previous_x, previous_y), (index_finger_x, index_finger_y), (233, 37, 0), brush_thickness)
                cv2.line(canvas, (previous_x, previous_y), (index_finger_x, index_finger_y), (233, 37, 0), brush_thickness)

                previous_x, previous_y = index_finger_x, index_finger_y

        # Combine canvas and live video feed
        cv2.imshow("Live Feed", frame)
        cv2.imshow("Drawing Canvas", canvas)

        cv2.waitKey(1)


if __name__ == "__main__":
    main()

