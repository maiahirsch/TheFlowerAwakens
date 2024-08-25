import cv2
import mediapipe as mp
import pyfirmata2
import time
import math

# Connect to the Arduino board
board = pyfirmata2.Arduino("/dev/cu.usbmodem11401")  # port where Arduino is connected

# Initialize the servo pin
servo_pin = board.get_pin("d:3:s")  # digital, pin 3, servo

# Start the iterator to avoid overflow
it = pyfirmata2.util.Iterator(board)
it.start()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand = mp_hands.Hands(max_num_hands=2)

# Initial angle for the servo
servo_angle = 90

# Smoothing parameters
smoothing_factor = 0.1  # Adjust this factor as needed

while True:
    success, frame = cap.read()
    if success:
        RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hand.process(RGB_frame)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]

                # Calculate the angle between thumb_tip and index_tip
                dx = index_tip.x - thumb_tip.x
                dy = index_tip.y - thumb_tip.y
                angle_rad = math.atan2(dy, dx)
                angle_deg = math.degrees(angle_rad)

                # Print the angle to the command window
                print(f"Angle between thumb and index tip: {angle_deg:.4f} degrees")

                # Normalize the angle to the range 0-180 for the servo
                normalized_angle = (angle_deg + 180) / 360 * 180

                # Smooth the servo angle
                servo_angle = (1 - smoothing_factor) * servo_angle + smoothing_factor * normalized_angle
                servo_pin.write(servo_angle)

                # Draw the hand landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Draw the line between thumb_tip and index_tip
                thumb_tip_coords = (int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0]))
                index_tip_coords = (int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0]))
                cv2.line(frame, thumb_tip_coords, index_tip_coords, (0, 255, 0), 2)  # Green line with thickness 2

        cv2.imshow('capture image', frame)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
board.exit()  # Clean up and close the connection to the Arduino
