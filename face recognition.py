import face_recognition
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os

path = 'known_faces'
images = []
classNames = []
for filename in os.listdir(path):
    img = cv2.imread(f'{path}/{filename}')
    images.append(img)
    classNames.append(os.path.splitext(filename)[0])

def find_encodings(images):
    encode_list = []
    for img in images:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img_rgb)
        if encodings:
            encode_list.append(encodings[0])
    return encode_list

known_encodings = find_encodings(images)

def mark_attendance(name):
    date_str = datetime.now().strftime('%Y-%m-%d')
    time_str = datetime.now().strftime('%H:%M:%S')
    file_name = f"Attendance_{date_str}.csv"
    if not os.path.exists(file_name):
        df = pd.DataFrame(columns=["Name", "Time"])
        df.to_csv(file_name, index=False)
    df = pd.read_csv(file_name)
    if name not in df['Name'].values:
        new_entry = pd.DataFrame([[name, time_str]], columns=["Name", "Time"])
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(file_name, index=False)

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    for encode_face, face_loc in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, encode_face)
        face_distances = face_recognition.face_distance(known_encodings, encode_face)
        match_index = np.argmin(face_distances)
        if matches[match_index]:
            name = classNames[match_index].upper()
            mark_attendance(name)
            y1, x2, y2, x1 = [v * 4 for v in face_loc]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Face Recognition Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
