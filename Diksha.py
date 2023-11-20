import cv2
import dlib
import numpy as np
import os

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("D:\Lip_Detection\shape_predictor_68_face_landmarks.dat~\shape_predictor_68_face_landmarks.dat")

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Flags to control buttons
close_camera = False
focus_on_lips = False
capture_lip_shape = False
image_counter = 1

# Lists to store lip landmarks for different shapes
lip_shapes_landmarks = []

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if focus_on_lips:
        # Use dlib to detect faces in the grayscale frame
        faces = detector(gray)

        for face in faces:
            # Predict facial landmarks
            landmarks = predictor(gray, face)

            # Extract the lip endpoints
            lip_endpoints = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)])

            # Calculate the lip area
            lip_area = cv2.contourArea(lip_endpoints)

            # Adjust the number of landmarks based on the lip area
            if lip_area < 10000:  # Adjust this threshold based on your observations
                selected_landmarks = list(range(48, 60))  # Fewer landmarks for smaller lips
            elif lip_area > 20000:
                selected_landmarks = list(range(48, 68))  # Full landmarks for larger lips
            else:
                selected_landmarks = list(range(48, 64))  # Intermediate landmarks for medium-sized lips

            # Draw the lip contour using selected landmarks
            lip_endpoints_selected = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in selected_landmarks])
            cv2.polylines(frame, [lip_endpoints_selected], True, (0, 255, 0), 2)

            if capture_lip_shape:
                # Store the captured landmarks in the list
                lip_shapes_landmarks.append(lip_endpoints_selected)

                # Reset flag after capturing
                capture_lip_shape = False

                # Print the captured lip landmarks
                print(f"Lip shape {image_counter} landmarks:")
                print(lip_endpoints_selected)
                image_counter += 1  # Increment the counter for the next image

    else:
        # Display the entire frame with the face highlighted
        cv2.imshow('Focused Face', frame)

    key = cv2.waitKey(1)

    if key & 0xFF == ord('q') or close_camera:
        break

    if key & 0xFF == ord('c'):
        close_camera = True

    if key & 0xFF == ord('s'):
        focus_on_lips = not focus_on_lips  # Toggle lip focus on 's' key press

    if key & 0xFF == ord('k'):
        capture_lip_shape = True  # Enable capturing lip shape when 'k' is pressed

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
