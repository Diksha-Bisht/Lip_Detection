import cv2
import dlib
import numpy as np
import os
import time

def extract_stats_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        mean_distance = float(lines[0].split(':')[-1].strip())
        variance_distance = float(lines[1].split(':')[-1].strip())
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


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"/Users/diksha_bisht/Desktop/Github-Sem-!/Lip_Detection/Model/shape_predictor_68_face_landmarks (1).dat")
successful_matches_count = 0

print("Press 'v' to start capturing live frames.")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
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

    # Lists to store the best similarity score for each frame
    best_similarity_scores = []

    while frame_count < 10:  # Capture live data for 10 frames
        time.sleep(0.5)

        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)
        live_landmarks = []

        for face in faces:
            landmarks = predictor(gray, face)
            lip_points_live = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)]
            live_landmarks.append(lip_points_live)

        # Paths to pre-stored landmark files
        stored_landmarks_paths = [
            f"/Users/diksha_bisht/Desktop/Github-Sem-!/Lip_Detection/dot/image_{i}/landmarks_{i}.txt"
            for i in range(1, 6)
        ]

        similarity_scores = []

        for stored_landmarks_path in stored_landmarks_paths:
            if not os.path.exists(stored_landmarks_path):
                print(f"Data file not found: {stored_landmarks_path}")
                continue

            stored_landmarks_distances = extract_landmarks_data(stored_landmarks_path)

            similarity_percentage = compare_landmarks(live_landmarks[0], stored_landmarks_distances)
            similarity_scores.append(similarity_percentage)

        if similarity_scores:
            best_similarity = max(similarity_scores)
            best_similarity_scores.append(best_similarity)

            print(f"Frame {frame_count + 1}: Best Similarity Score: {best_similarity:.2f}%")

            if best_similarity > 80:
                success_count += 1

        frame_count += 1

    if best_similarity_scores:
        average_similarity = sum(best_similarity_scores) / len(best_similarity_scores)
        print(f"Average Best Similarity across frames: {average_similarity:.2f}%")

    if success_count >= 8:
        print("Success")
    else:
        print("Failure")

    cv2.destroyAllWindows()
    break