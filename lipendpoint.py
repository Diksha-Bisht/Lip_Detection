# Import necessary libraries
import cv2
import dlib
import os
import numpy as np

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

# Specify the folder path where lip images will be stored
output_folder = 'D:\Lip_Detection\lip\shape'  # Change this to your desired output folder

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

            # Extract the lip endpoints (you might need to adjust the landmark indices)
            lip_endpoints = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)])

            # Draw the lip contour on the frame
            cv2.polylines(frame, [lip_endpoints], True, (0, 255, 0), 2)

            if capture_lip_shape:
                # Crop the lip region based on the detected endpoints
                min_x = np.min(lip_endpoints[:, 0])
                max_x = np.max(lip_endpoints[:, 0])
                min_y = np.min(lip_endpoints[:, 1])
                max_y = np.max(lip_endpoints[:, 1])
                lips_region = frame[min_y:max_y, min_x:max_x]

                if lips_region.any():
                    cv2.imshow('Lips Region', lips_region)

                    # Save the captured lip region with a unique name when 'k' key is pressed
                    key = cv2.waitKey(1)
                    if key & 0xFF == ord('k'):
                        file_name = f'lip_shape_{image_counter}.png'
                        file_path = os.path.join(output_folder, file_name)
                        cv2.imwrite(file_path, lips_region)
                        print(f'Lip shape {image_counter} has been saved as {file_path}')
                        capture_lip_shape = False  # Reset flag after capturing
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
