import cv2
import dlib
import numpy as np
import os


def compare_live_stats(live_stats, stored_stats):
    if len(live_stats) < 2:
        print("Live stats tuple is empty or contains only one element. Skipping comparison.")
        return 0

    live_mean = live_stats[0]
    live_variance = live_stats[1]
    stored_mean = stored_stats[0]
    stored_variance = stored_stats[1]

    # Calculate the absolute difference between live and stored mean distances
    mean_difference = abs(live_mean[0] - stored_mean)

    # Calculate the absolute difference between live and stored variances
    variance_difference = abs(live_variance[1] - stored_variance)

    # Normalize the differences to a range of 0 to 1
    normalized_mean_difference = mean_difference / (max(live_mean[0], stored_mean) - min(live_mean[0], stored_mean))
    normalized_variance_difference = variance_difference / (max(live_variance[1], stored_variance) - min(live_variance[1], stored_variance))

    # Combine the normalized differences into a single similarity score
    stats_similarity_score = (1 - normalized_mean_difference) * (1 - normalized_variance_difference) * 100

    # Ensure the similarity score is between 1 and 100
    stats_similarity_score = max(1, min(stats_similarity_score, 100))

    return stats_similarity_score


def extract_stats_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        mean_distance = float(lines[0].split(':')[-1].strip())
        variance_distance = float(lines[1].split(':')[-1].strip())
    return mean_distance, variance_distance


def main():
    # Define paths for stored lip stats files
    stored_stats_paths = [
        f"dot/image_{i}/lips_stats_{i}.txt"
        for i in range(1, 14)
    ]

    # Capture live frames for 10 frames
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(r"Model/shape_predictor_68_face_landmarks (1).dat")
    cap = cv2.VideoCapture(0)

    frame_count = 1
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('v'):
            # Detect faces and extract lip landmarks
            faces = detector(gray)
            live_landmarks = []
            for face in faces:
                landmarks = predictor(gray, face)
                lip_points_live = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)]
                live_landmarks.append(lip_points_live)

            # Calculate mean distance and variance
                distances = [calculate_distance(p1, p2) for p1 in lip_points for p2 in lip_points if lip_points.index(p1) < lip_points.index(p2)]
                mean_distance = np.mean(distances)
                variance_distance = np.var(distances)
                
            # Compare live stats to stored stats for each face
            for live_face_landmarks in live_landmarks:
                best_similarity = 0
                for stored_stats_path in stored_stats_paths:
                    if not os.path.exists(stored_stats_path):
                        continue

                    stored_stats = extract_stats_data(stored_stats_path)
                    similarity_score = compare_live_stats(live_face_landmarks, stored_stats)
                    best_similarity = max(best_similarity, similarity_score)

                print(f"Frame {frame_count} Similarity Score: {best_similarity}")

        # Display the frame
        cv2.imshow('Live Lip Landmark Detection', frame)

        frame_count += 1

        # Break the loop if 'q' key is pressed
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
