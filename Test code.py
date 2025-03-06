import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Load the model
path = r"C:\Users\lenovo\OneDrive - mgmtech\Documents\MIT WPU\4. FY\Capstone\Github\Capstone\model_inception.h5"
converted_path = path.replace("\\", "/")
model_path = converted_path

try:
    model = load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Define the input size expected by the model
input_size = (224, 224)  # Adjust if necessary

# Original class mapping
original_classes = {
    '(BT) Body Tissue or Organ': 0,
    '(GE) Glass equipment-packaging 551': 1,
    '(ME) Metal equipment -packaging': 2,
    '(OW) Organic wastes': 3,
    '(PE) Plastic equipment-packaging': 4,
    '(PP) Paper equipment-packaging': 5,
    '(SN) Syringe needles': 6,
    'Gauze': 7,
    'Gloves': 8,
    'Mask': 9,
    'Syringe': 10,
    'Tweezers': 11
}

# New mapping to broader categories
broad_categories = {
    'infectious waste': ['Gauze', 'Gloves', 'Mask', '(PE) Plastic equipment-packaging', '(PP) Paper equipment-packaging'],
    'pathological waste': ['(BT) Body Tissue or Organ'],
    'sharps': ['(SN) Syringe needles', 'Syringe', 'Tweezers']
}

# Create a mapping from original class indices to broad categories
category_mapping = {}
for category, items in broad_categories.items():
    for item in items:
        if item in original_classes:
            category_mapping[original_classes[item]] = category

# Function to get the broad category
def get_broad_category(predicted_class_index):
    return category_mapping.get(predicted_class_index, "Unknown")

# Function to preprocess and predict an image
def predict_image(img):
    img = cv2.resize(img, input_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_prob = predictions[0][predicted_class_index]
    return predicted_class_index, predicted_class_prob

# Function to display prediction results
def display_prediction(frame, predicted_class_index, predicted_class_prob):
    broad_category = get_broad_category(predicted_class_index)
    cv2.putText(frame, f'Category: {broad_category} (Confidence: {predicted_class_prob:.2f})', 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Prediction', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main menu
def main_menu():
    print("Choose an option:")
    print("1. Real-time prediction (live video)")
    print("2. Capture and predict (take a picture)")
    print("3. Predict from local image")
    print("4. Exit")
    choice = input("Enter your choice (1/2/3/4): ")
    return choice

# Real-time prediction
def real_time_prediction():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Predict the class and confidence
        predicted_class_index, predicted_class_prob = predict_image(frame)

        # Display the prediction on the live feed
        if predicted_class_prob >= 0.7:  # Confidence threshold
            broad_category = get_broad_category(predicted_class_index)
            cv2.putText(frame, f'Category: {broad_category} (Confidence: {predicted_class_prob:.2f})', 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No confident prediction", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the live feed
        cv2.imshow('Real-time Prediction', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Capture and predict
def capture_and_predict():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        return

    predicted_class_index, predicted_class_prob = predict_image(frame)
    if predicted_class_prob >= 0.7:  # Confidence threshold
        display_prediction(frame, predicted_class_index, predicted_class_prob)

    cap.release()

# Predict from local image
def predict_from_local_image():
    Temp = r"C:\Users\lenovo\OneDrive - mgmtech\Documents\MIT WPU\4. FY\Capstone\Github\Capstone\Test Images"
    converted_folder_path = Temp.replace("\\", "/")
    folder_path = converted_folder_path
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return

    images = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        print(f"No images found in '{folder_path}'.")
        return

    print(f"Found {len(images)} images in '{folder_path}'. Processing all images...")

    for image_name in images:
        image_path = os.path.join(folder_path, image_name)
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not read image '{image_path}'.")
            continue

        # Predict the class and confidence
        predicted_class_index, predicted_class_prob = predict_image(frame)

        # Display the prediction
        if predicted_class_prob >= 0.7:  # Confidence threshold
            broad_category = get_broad_category(predicted_class_index)
            print(f"Image: {image_name} | Category: {broad_category} | Confidence: {predicted_class_prob:.2f}")
            cv2.putText(frame, f'Category: {broad_category} (Confidence: {predicted_class_prob:.2f})', 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            print(f"Image: {image_name} | No confident prediction (Confidence: {predicted_class_prob:.2f})")
            cv2.putText(frame, "No confident prediction", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the image with prediction
        cv2.imshow('Prediction', frame)
        cv2.waitKey(0)  # Wait for a key press to move to the next image

    cv2.destroyAllWindows()

# Main program loop
while True:
    choice = main_menu()
    if choice == '1':
        real_time_prediction()
    elif choice == '2':
        capture_and_predict()
    elif choice == '3':
        predict_from_local_image()
    elif choice == '4':
        print("Exiting...")
        break
    else:
        print("Invalid choice. Please try again.")