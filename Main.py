import cv2
import numpy as np
from tensorflow.keras.models import load_model
import serial  # Import serial library for communication with Arduino
import time

# Load the model
model_path = r"C:/Users/lenov\OneDrive - mgmtech\Documents\MIT WPU\4. FY\Capstone\Github\Capstone\model_inception.h5"
model = load_model(model_path)

# Define the input size expected by the model
input_size = (224, 224)  # Adjust if necessary

# List of color labels based on your dataset
color_labels = [
    'Red',
    'Blue',
    'Green',
    'Yellow'
]

# Attempt to open the external webcam (typically index 1 or higher for external cams)
cap = cv2.VideoCapture(1)  # Change to 1 or the appropriate index for your external camera

if not cap.isOpened():
    print("Error: Could not access webcam.")
    exit()

# Set up serial communication with Arduino
arduino_port = 'COM5'  # Replace with your Arduino's port (e.g., /dev/ttyUSB0 for Linux)
baud_rate = 9600
ser = serial.Serial(arduino_port, baud_rate)
time.sleep(2)  # Wait for Arduino to initialize

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Preprocess the frame
    img = cv2.resize(frame, input_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Make predictions
    predictions = model.predict(img)

    # Get the predicted class index and its probability
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_prob = predictions[0][predicted_class_index]

    # Get the class label
    predicted_label = color_labels[predicted_class_index]

    # Display the prediction result on the frame
    cv2.putText(frame, f'Prediction: {predicted_label} (Confidence: {predicted_class_prob:.2f})', 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Object Detection', frame)

    # Send the predicted label to Arduino via serial
    if predicted_label == 'Red':
        ser.write(b'R')  # Send 'R' for Red
    elif predicted_label == 'Blue':
        ser.write(b'B')  # Send 'B' for Blue
    elif predicted_label == 'Green':
        ser.write(b'G')  # Send 'G' for Green
    elif predicted_label == 'Yellow':
        ser.write(b'Y')  # Send 'Y' for Yellow

    # Break the loop on 'q' key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Exiting...")
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

# Close the serial connection
ser.close()