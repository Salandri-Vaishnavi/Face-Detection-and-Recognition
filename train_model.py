
import cv2
import os
import numpy as np

# Dataset and model directories
dataset_dir = "dataset"
model_dir = "models"

os.makedirs(model_dir, exist_ok=True)

# Selected identities for training
selected_people = [
    "vaishnavi",
    "Krishna",
    "Virat Kohli"
]

# Check LBPH availability
if not hasattr(cv2, "face") or not hasattr(
    cv2.face, "LBPHFaceRecognizer_create"
):
    raise RuntimeError(
        "LBPH is unavailable. Check your OpenCV installation."
    )

# Initialize recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []

label_to_name = {}
current_label = 0

# Check dataset directory
if not os.path.isdir(dataset_dir):
    raise FileNotFoundError("Dataset folder not found.")

# Load only selected people
for person_name in selected_people:

    person_path = os.path.join(dataset_dir, person_name)

    if not os.path.isdir(person_path):
        print(f"Folder not found: {person_name}")
        continue

    person_images = []

    for filename in sorted(os.listdir(person_path)):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        image_path = os.path.join(person_path, filename)

        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        image = cv2.resize(image, (200, 200))
        person_images.append(image)

    if not person_images:
        print(f"No valid images found for {person_name}.")
        continue

    label_to_name[current_label] = person_name

    for image in person_images:
        faces.append(image)
        labels.append(current_label)

    print(
        f"Loaded {len(person_images)} images for {person_name}."
    )

    current_label += 1

# Check training data
if not faces:
    raise RuntimeError("No training images found.")

# Train recognizer
recognizer.train(
    faces,
    np.array(labels, dtype=np.int32)
)

# Save model
model_path = os.path.join(
    model_dir, "face_model.yml"
)

recognizer.save(model_path)

# Save labels
names_path = os.path.join(
    model_dir, "labels.txt"
)

with open(names_path, "w", encoding="utf-8") as file:
    for label, name in label_to_name.items():
        file.write(f"{label}:{name}\n")

print("\nModel training completed successfully!")
print(f"Total training images: {len(faces)}")
print(f"Registered people: {len(label_to_name)}")
print(f"Trained model saved to: {model_path}")
print(f"Label mapping saved to: {names_path}")
