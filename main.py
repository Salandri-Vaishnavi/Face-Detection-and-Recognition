
import cv2
import os

# Load the face detection model
cascade_path = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(cascade_path)

# Load the trained face recognition model
model_path = "models/face_model.yml"
labels_path = "models/labels.txt"

if not os.path.exists(model_path):
    raise FileNotFoundError(
        "Model not found. Run train_model.py first."
    )

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)

# Load the label mapping


labels = {}

with open(labels_path, "r") as file:
    for line in file:
        line = line.strip()

        if not line:
            continue

        label, name = line.split(":", maxsplit=1)
        labels[int(label)] = name




# Start the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    raise SystemExit

print("Face recognition started!")
print("Press Q to quit.")

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Could not read webcam frame.")
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (200, 200))

            label, confidence = recognizer.predict(face_roi)

            # Lower confidence distance means a closer match
            if label in labels and confidence < 70:
                name = labels[label]
            else:
                name = "Unknown"

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                color,
                2
            )

            cv2.putText(
                frame,
                f"{name} ({confidence:.1f})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        cv2.imshow("Face Recognition System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
