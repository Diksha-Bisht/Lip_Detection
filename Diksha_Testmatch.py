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
        distances = [float(line.split(':')[-1].strip()) for line in lines[1:]]
    return distances

def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def compare_landmarks(live_landmarks, stored_landmarks):
    similarity_score = 0
    for live_point, stored_point in zip(live_landmarks, stored_landmarks):
        similarity_score += calculate_distance(live_point, stored_point)
    normalized_similarity_score = similarity_score / len(live_landmarks) * 100
    return normalized_similarity_score

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"D:\Lip_Detection\Model\shape_predictor_68_face_landmarks (1).dat")

successful_matches_count = 0
similarity_scores = []

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
        cap.release()
        cv2.destroyAllWindows()
        exit()

base_folder = r'D:\Lip_Detection\dot'

for image_folder_number in range(1, 6):  # Loop through 5 image folders
    image_folder_path = os.path.join(base_folder, f'image_{image_folder_number}')

    if not os.path.exists(image_folder_path):
        print(f"Image folder {image_folder_number} not found")
        continue

    for file_number in range(1, 6):  # Loop through 5 stored files in each image folder
        stats_data_path = os.path.join(image_folder_path, f'lips_stats_{file_number}.txt')
        landmarks_data_path = os.path.join(image_folder_path, f'lips_{file_number}.txt')

        if not os.path.exists(stats_data_path) or not os.path.exists(landmarks_data_path):
            print(f"Data files not found for stored frame {file_number} in image folder {image_folder_number}")
            continue

        # Extracting mean distance and variance from stats file
        with open(stats_data_path, 'r') as stats_file:
            lines = stats_file.readlines()
            mean_distance = float(lines[1].split(':')[-1].strip())

            if len(lines) >= 3:
                variance_distance = float(lines[2].split(':')[-1].strip())
            else:
                variance_distance = 0  # Set a default value if variance is not available

        # Extracting landmark distances from landmarks file
        with open(landmarks_data_path, 'r') as landmarks_file:
            lines = landmarks_file.readlines()
            stored_landmarks_distances = [float(line.split(':')[-1].strip()) for line in lines[1:]]

        frame_count = 0
        similarity_scores = []

        while frame_count < 5:  # Capture live data for 5 frames
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

            similarity_percentage = compare_landmarks(live_landmarks, stored_landmarks_distances)
            print(f"Similarity Score for image folder {image_folder_number}, frame {file_number}: {similarity_percentage:.2f}%")
            frame_count += 1

            similarity_scores.append(similarity_percentage)
            if similarity_percentage >= mean_distance:  # Set your threshold here
                successful_matches_count += 1

print(f"Total Successful Matches: {successful_matches_count}/{len(similarity_scores)}")
print("Exiting...")
cap.release()
cv2.destroyAllWindows()
