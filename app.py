from flask import Flask, render_template, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/update', methods=['GET', 'POST'])
def update():
    import cv2
    import dlib
    import os
    import numpy as np
    import tkinter as tk

    # Create a window to get the screen dimensions
    root = tk.Tk()

    # Get the screen width and height
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Destroy the window as it's not needed anymore
    root.destroy()

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
        major_axis_percentage = 8  # Percentage of screen width for major axis
        minor_axis_percentage = 18  # Percentage of screen height for minor axis
        # Convert percentages to actual dimensions
        major_axis = int((major_axis_percentage / 100) * screen_width)
        minor_axis = int((minor_axis_percentage / 100) * screen_height)
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


@app.route('/login', methods=['GET', 'POST'])

def login():
    import cv2
    import dlib
    import numpy as np
    import os
    import time

    def extract_stats_data(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            mean_distance = float(lines[0].strip())
            variance_distance = float(lines[1].strip())
        return mean_distance, variance_distance

    def extract_landmarks_data(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            landmarks = [
                tuple(map(int, line.strip().split(',')))
                for line in lines
            ]
        return landmarks

    def calculate_distance(point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def compare_landmarks(live_landmarks, stored_landmarks):
        max_landmark_distance = 100  # Assuming the maximum distance between landmarks is 100

        # Calculate the sum of squared distances
        sum_squared_distances = 0
        for live_point, stored_point in zip(live_landmarks, stored_landmarks):
            squared_distance = calculate_distance(live_point, stored_point)**2
            sum_squared_distances += squared_distance

        # Calculate the root mean square error (RMSE)
        rmse = np.sqrt(sum_squared_distances / len(live_landmarks))

        # Normalize the RMSE to a range of 0 to 1
        normalized_similarity_score = 1 - rmse / (2 * max_landmark_distance)

        # Convert the normalized score to a percentage
        percentage_similarity_score = normalized_similarity_score * 100

        # Ensure the similarity score is between 1 and 100
        percentage_similarity_score = max(1, min(percentage_similarity_score, 100))

        return percentage_similarity_score

    def compare_live_stats(live_distances, stored_stats):
        # Similarity calculation between live and stored stats
        stored_mean_distance, stored_variance_distance = stored_stats

        live_mean_distance = np.mean(live_distances)
        live_variance_distance = np.var(live_distances)

        # Calculate similarity for mean and variance separately
        mean_similarity = 100 * (1 - abs(stored_mean_distance - live_mean_distance) / stored_mean_distance)
        variance_similarity = 100 * (1 - abs(stored_variance_distance - live_variance_distance) / stored_variance_distance)

        # Average similarity of mean and variance for final stats similarity
        stats_similarity_percentage = (mean_similarity + variance_similarity) / 2
        return stats_similarity_percentage

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(r"Model/shape_predictor_68_face_landmarks (1).dat")

    # Function to draw an ellipse on the frame
    def draw_ellipse(frame, center, major_axis, minor_axis, angle):
        ellipses_axes = (major_axis, minor_axis)  # Combine major and minor axis into a tuple
        cv2.ellipse(frame, center, ellipses_axes, angle, 0, 360, (255, 0, 0), 1)

    print("Press 'v' to start capturing live frames.")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
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
                cv2.circle(frame, point, 1, (0, 255, 0), -1)
        
        cv2.imshow('Press V to Start', frame)
        
        key = cv2.waitKey(1)
        if key == ord('v'):
            print("Comparison started...")
            break
        elif key == 27:
            print("Exiting...")
            exit()

    while True:
        frame_count = 0
        success_count = 0

        while frame_count < 10:  # Capture live data for 10 frames
            time.sleep(0.5)

            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = detector(gray)
            live_landmarks = []

            for face in faces:
                landmarks = predictor(gray, face)
                lip_points_live = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)]
                live_landmarks.append(lip_points_live)

            # Calculate distances for live stats
            distances = [calculate_distance(p1, p2) for p1 in lip_points for p2 in lip_points if lip_points.index(p1) < lip_points.index(p2)]

            # Similarity for landmarks
            # Paths to pre-stored landmark files
            stored_landmarks_paths = [
                f"dot/image_{i}/landmarks_{i}.txt"
                for i in range(1, 14)
            ]
            similarity_scores = []

            for stored_landmarks_path in stored_landmarks_paths:
                if not os.path.exists(stored_landmarks_path):
                    print(f"Data file not found: {stored_landmarks_path}")
                    continue

                # Extracting and comparing live landmarks
                stored_landmarks_distances = extract_landmarks_data(stored_landmarks_path)
                similarity_percentage = compare_landmarks(live_landmarks[0], stored_landmarks_distances)
                similarity_scores.append(similarity_percentage)

            # Similarity for Stats
            # Paths for stored stats files
            stored_stats_paths = [
                f"dot/image_{i}/lips_stats_{i}.txt"
                for i in range(1, 14)
            ]
            best_stats_similarity = 0

            for stored_stats_path in stored_stats_paths:
                if not os.path.exists(stored_stats_path):
                    print(f"Data file not found: {stored_stats_path}")
                    continue

                # Extracting and comparing live stats
                stored_stats = extract_stats_data(stored_stats_path)
                stats_similarity_percentage = compare_live_stats(distances, stored_stats)
                best_stats_similarity = max(best_stats_similarity, stats_similarity_percentage)

            if similarity_scores:
                best_similarity = max(similarity_scores)
                print(f"Frame {frame_count + 1}: Landmarks Similarity: {best_similarity:.2f}%, Stats Similarity: {best_stats_similarity:.2f}%")

                if best_similarity > 90 and best_stats_similarity > 90:
                    success_count += 1

            frame_count += 1

        if success_count >= 8:
            print("Success")
        else:
            print("Failure")

        return jsonify(success_count=success_count)

if __name__ == '__main__':
    app.run(debug=True)
