
import cv2
import os
import re

# Get person's name
name = input("Enter the person's name: ").strip()

if not name or not re.fullmatch(r"[A-Za-z0-9 ._-]+", name):
    print("Please enter a valid name.")
    raise SystemExit

# Create folder
save_path = os.path.join("dataset", name)
os.makedirs(save_path, exist_ok=True)

# Count existing images
count = len([
    f for f in os.listdir(save_path)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

# Load Haar Cascade
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print("Face detector loading failed.")
    raise SystemExit

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera could not open.")
    raise SystemExit

print("\nCamera started successfully!")
print("Click the webcam window and press S to save.")
print("Press Q to finish registration.\n")

while True:

    success, frame = camera.read()

    if not success:
        print("Unable to read webcam.")
        break

    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    cv2.putText(
        frame,
        f"Saved Images: {count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("Face Registration", frame)

    # Keyboard input
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') or key == ord('S'):
        print("S KEY PRESSED")
        print("Number of detected faces:", len(faces))

    if key == ord('s') or key == ord('S'):
        print("S KEY PRESSED")
        print("Detected faces:", len(faces))

        # Detect face again from the current frame
        current_faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(60, 60)
        )

        if len(current_faces) == 0:
            print("No face detected. Try again.")
            continue

        # Select largest face
        x, y, w, h = max(
            current_faces,
            key=lambda face: face[2] * face[3]
        )

        # Crop face
        face = gray[y:y+h, x:x+w]

        # Resize
        face = cv2.resize(face, (200, 200))

        # Create unique filename
        count += 1
        filename = os.path.join(
            save_path,
            f"{name}_{count}.jpg"
        )

        # Save image
        result = cv2.imwrite(filename, face)

        if result:
            print(f"SUCCESS: Image saved - {filename}")
        else:
            count -= 1
            print("ERROR: Image could not be saved.")

    elif key == ord('q') or key == ord('Q'):
        break

camera.release()
cv2.destroyAllWindows()

print("\nRegistration completed!")
print("Total images saved:", count)
print("Folder location:", os.path.abspath(save_path))
