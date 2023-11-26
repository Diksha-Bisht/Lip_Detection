def update():
    import cv2
    import dlib
    import os
    import numpy as np

    # Initialize dlib's face detector and shape predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("Model/shape_predictor_68_face_landmarks (1).dat")

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Specify the folder path where lip images will be stored
    output_folder = 'dot'  # Change this to your desired output folder

    # Flags to control capture
    capture_lip_shape = False
    image_counter = 1

    # Function to calculate distance between two points
    def calculate_distance(point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    # Function to draw an ellipse on the frame
    def draw_ellipse(frame, center, major_axis, minor_axis, angle):
        ellipses_axes = (major_axis, minor_axis)  # Combine major and minor axis into a tuple
        cv2.ellipse(frame, center, ellipses_axes, angle, 0, 360, (255, 0, 0), 2)


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Define ellipse parameters
        center = (frame.shape[1] // 2, frame.shape[0] // 2)
        major_axis = 200  # Length of the major axis
        minor_axis = 250  # Length of the minor axis
        angle = 0  # Angle of rotation in degrees
        # Draw the ellipse on the frame
        draw_ellipse(frame, center, major_axis, minor_axis, angle)

        # Detect faces
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            lip_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)]
            for point in lip_points:
                cv2.circle(frame, point, 1, (0, 0, 255), -1)
        
        key = cv2.waitKey(1)

        if capture_lip_shape:
            # Capture lip shape when 's' key is pressed
            for face in faces:
                landmarks = predictor(gray, face)
                lip_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)]

                # Create a folder for the current image's lip data
                folder_path = os.path.join(output_folder, f'image_{image_counter}')
                os.makedirs(folder_path, exist_ok=True)

                # Save distances to separate text files for each image
                landmarks_file_path = os.path.join(folder_path, f'landmarks_{image_counter}.txt')
                with open(landmarks_file_path, 'w') as file:
                    for point in lip_points:
                        file.write(f"{point[0]}, {point[1]}\n")
                
                # Calculate mean distance and variance
                distances = [calculate_distance(p1, p2) for p1 in lip_points for p2 in lip_points if lip_points.index(p1) < lip_points.index(p2)]
                mean_distance = np.mean(distances)
                variance_distance = np.var(distances)

                # Save mean and variance to a text file
                stats_file_path = os.path.join(folder_path, f'lips_stats_{image_counter}.txt')
                with open(stats_file_path, 'w') as stats_file:
                    stats_file.write(f"{mean_distance}\n")
                    stats_file.write(f"{variance_distance}\n")
                
                print(f'Lip distances stats {image_counter} saved in {folder_path}')

                capture_lip_shape = False  # Reset flag
                image_counter += 1  # Increment image counter

        cv2.imshow('Lip Landmarks', frame)

        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('s'):
            capture_lip_shape = True  # Capture lip shape on 's' key press

    cap.release()
    cv2.destroyAllWindows()



update()