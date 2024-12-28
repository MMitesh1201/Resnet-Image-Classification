# inference.py
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import load_model

def preprocess_image(img):
    """
    Preprocess an image for ResNet50V2 CIFAR-10 prediction.
    
    Args:
    - img: Input image (BGR format from cv2)
    
    Returns:
    - Preprocessed image ready for model prediction
    """
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to 32x32 (CIFAR-10 size)
    img = cv2.resize(img, (32, 32))
    
    # Normalize pixel values
    img = img.astype('float32') / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

def capture_and_classify():
    """
    Capture a single image from webcam and classify it.
    Press SPACE to capture, ESC to quit without capturing.
    """
    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck']
    
    try:
        # Load the trained model
        model = load_model('resnet50v2_cifar10.h5')
    except:
        print("Error: Could not find the model file 'resnet50v2_cifar10.h5'")
        print("Please ensure you have trained the model first using train.py")
        return
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Press SPACE to capture an image or ESC to quit")
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Display the frame
        cv2.imshow('Press SPACE to capture', frame)
        
        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        
        # If ESC is pressed, quit without capturing
        if key == 27:  # ESC key
            print("Exiting without capture")
            break
            
        # If SPACE is pressed, process the image
        if key == 32:  # SPACE key
            # Get frame dimensions
            height, width = frame.shape[:2]
            
            # Preprocess the frame
            processed_frame = preprocess_image(frame)
            
            # Make prediction
            prediction = model.predict(processed_frame, verbose=0)
            predicted_class = class_names[np.argmax(prediction[0])]
            confidence = np.max(prediction[0]) * 100
            
            # Add prediction text to frame
            text = f"{predicted_class}: {confidence:.1f}%"
            result_frame = frame.copy()
            cv2.putText(result_frame, text, (10, height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Display the result
            cv2.imshow('Classification Result', result_frame)
            print(f"\nPrediction: {predicted_class}")
            print(f"Confidence: {confidence:.1f}%")
            
            # Wait for a key press before closing
            cv2.waitKey(0)
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_classify()