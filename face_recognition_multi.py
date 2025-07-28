import cv2
import os
import numpy as np

dataset_path = "dataset"
os.makedirs(dataset_path, exist_ok=True)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# STEP 1: CAPTURE DATA FOR MULTIPLE PEOPLE

while True:
    person_name = input("Enter name of person (or press Enter to finish): ").strip()
    if person_name == "":
        break  

    save_path = os.path.join(dataset_path, person_name)
    os.makedirs(save_path, exist_ok=True)
    print(f"[INFO] Capturing images for {person_name}. Press 'q' when done...")

    cap = cv2.VideoCapture(0)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            cv2.imwrite(f"{save_path}/{count}.jpg", face_img)
            count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow(f'Capturing - {person_name}', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Captured {count} images for {person_name}")

print("[INFO] Finished capturing data for all people.")

# STEP 2: TRAIN THE MODEL

print("[INFO] Training model...")
labels = []
faces_data = []
label_id = 0
names = {}

for person in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person)
    if not os.path.isdir(person_folder):
        continue
    names[label_id] = person
    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        faces_data.append(img)
        labels.append(label_id)
    label_id += 1

labels = np.array(labels)
model = cv2.face.LBPHFaceRecognizer_create()
model.train(faces_data, labels)
model.save("face_model.yml")
print("[INFO] Model trained and saved as face_model.yml")

# STEP 3: REAL-TIME RECOGNITION

print("[INFO] Starting real-time recognition. Press 'q' to quit.")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        label, confidence = model.predict(face)

        if confidence < 80:
            name = names[label]
            color = (0, 255, 0)
        else:
            name = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Recognition stopped. Exiting.")
