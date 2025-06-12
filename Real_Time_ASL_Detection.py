import cv2
import numpy as np
from tensorflow.keras.models import load_model

print("🚀 Script started")


MODEL_PATH = 'asl_transfer_model.h5'  
IMG_SIZE = 224
class_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
]

try:
    print("  Loading model...")
    model = load_model(MODEL_PATH)
    print(" Model loaded successfully!")
except Exception as e:
    print(f" Model load error: {e}")
    exit()

# Initialize webcam
print(" Starting webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print(" Cannot open webcam!")
    exit()
print(" Webcam started successfully!")

print("\n Instructions:")
print("- Show your hand gesture to the camera")
print("- Press 'q' to quit\n")

while True:
    print(" Capturing frame...")
    ret, frame = cap.read()
    if not ret:
        print(" Failed to grab frame")
        break

    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)

    # Preprocess frame for model
    try:
        img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
    except Exception as e:
        print(f" Preprocessing error: {e}")
        continue 
    try:
        preds = model.predict(img, verbose=0)
        pred_idx = np.argmax(preds)
        pred_label = class_names[pred_idx]
        confidence = np.max(preds)
    except Exception as e:
        print(f" Prediction error: {e}")
        pred_label = "Error"
        confidence = 0

   
    text = f"{pred_label} ({confidence:.2f})"
    cv2.putText(frame, text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    
    cv2.imshow('ASL Real-Time Detection', frame)

    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print(" Quitting...")
        break

cap.release()
cv2.destroyAllWindows()
print("  Detection stopped. Goodbye!")