import cv2
import dlib
import os

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("D:\Lip_Detection\Model\shape_predictor_68_face_landmarks (1).dat")  # Change to your predictor file path

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Specify the folder path where lip images will be stored
output_folder = 'D:\Lip_Detection\lip_dataset'  # Change this to your desired output folder

# Flags to control capture and lip display
capture_lip_shape = False
focus_on_lips = False
image_counter = 1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)

    for face in faces:
        # Capture lip shape
        if capture_lip_shape:
            landmarks = predictor(gray, face)
            lip_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)]

            # Crop lip region
            min_x = min(lip_points, key=lambda x: x[0])[0]
            max_x = max(lip_points, key=lambda x: x[0])[0]
            min_y = min(lip_points, key=lambda x: x[1])[1]
            max_y = max(lip_points, key=lambda x: x[1])[1]
            lips_region = frame[min_y:max_y, min_x:max_x]

            # Save the lip region
            file_name = f'lip_shape_{image_counter}.png'
            file_path = os.path.join(output_folder, file_name)
            cv2.imwrite(file_path, lips_region)
            print(f'Lip shape {image_counter} has been saved as {file_path}')

            capture_lip_shape = False  # Reset flag
            image_counter += 1  # Increment image counter

    if focus_on_lips:
        # Display lip landmarks
        for face in faces:
            landmarks = predictor(gray, face)
            lip_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)]
            for point in lip_points:
                cv2.circle(frame, point, 2, (0, 255, 0), -1)

    cv2.imshow('Lip Landmarks', frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('s'):
        focus_on_lips = not focus_on_lips  # Display lip landmarks on 's' key press
    elif key & 0xFF == ord('k'):
        capture_lip_shape = True  # Capture lip shape on 'k' key press

cap.release()
cv2.destroyAllWindows()
