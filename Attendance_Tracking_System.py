import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

def initialize_known_faces():
    """Load known faces and their encodings."""
    known_faces = []
    known_names = []

    # Add known faces and their names here

    known_faces.append(face_recognition.face_encodings(face_recognition.load_image_file("photos/pr.jpg"))[0])
    known_names.append("Lakshmi Narasimha Patnaik")

    return known_faces, known_names

def mark_attendance(name, entry=True):
    """Mark attendance in the CSV file."""
    file_exists = os.path.isfile('Attendance.csv')
    current_date = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime('%H:%M:%S')

    # Read existing entries
    rows = []
    if file_exists:
        with open('Attendance.csv', 'r', newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)

    # Check if header needs to be written
    if not file_exists or (file_exists and len(rows) == 0):
        with open('Attendance.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['NAME', 'DATE', 'ENTRY_TIME', 'EXIT_TIME'])

    # Check if the name already has an entry for the current date
    if entry:
        for row in rows:
            if row[0] == name and row[1] == current_date and row[2] != '':
                print(f"{name} has already entered today.")
                return  # Exit the function if entry already exists
        
        # If no entry exists, add a new row
        with open('Attendance.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, current_date, time_str, ''])

    # If exit, update the existing row
    else:
        updated = False
        for row in rows:
            if row[0] == name and row[1] == current_date and row[3] == '':
                row[3] = time_str
                updated = True
                break

        # Write back updated rows
        if updated:
            with open('Attendance.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(rows)

def main(entry):
    """Main function to run the face recognition attendance system."""
    video_capture = cv2.VideoCapture(0)
    known_faces, known_names = initialize_known_faces()

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture image")
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            face_distances = face_recognition.face_distance(known_faces, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_names[best_match_index]
                if entry:
                    mark_attendance(name, entry=True)
                else:
                    mark_attendance(name, entry=False)

                # Display name on the frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, name, (10, 50), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow("Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("For Entry-ENTER '1'\nFor Exit-ENTER '0'")
    entry = int(input())
    main(entry)




