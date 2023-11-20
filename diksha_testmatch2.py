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

        stats_data_path = rf'D:\Lip_Detection\dot\image_1\lips_stats_{frame_count + 1}.txt'
        landmarks_data_path = rf'D:\Lip_Detection\dot\image_1\lips_{frame_count + 1}.txt'
        print(stats_data_path)
        print(landmarks_data_path)

        if not os.path.exists(stats_data_path) or not os.path.exists(landmarks_data_path):
            print(f"Data files not found for frame {frame_count + 1}")
            frame_count += 1
            continue


        mean_distance, variance_distance = extract_stats_data(stats_data_path)
        stored_landmarks_distances = extract_landmarks_data(landmarks_data_path)

        similarity_percentage = compare_landmarks(live_landmarks[0], stored_landmarks_distances)
        similarity_scores.append(similarity_percentage)

        print(f"Frame {frame_count + 1}: Similarity Score: {similarity_percentage:.2f}%")
        frame_count += 1

    if len(similarity_scores) > 0:
        average_similarity = sum(similarity_scores) / len(similarity_scores)
        print(f"Average Similarity across frames: {average_similarity:.2f}%")

    cv2.destroyAllWindows()
    break
