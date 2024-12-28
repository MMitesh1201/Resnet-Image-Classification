import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

class ObjectClassifier:
    def __init__(self):
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck']
        try:
            self.model = load_model('resnet50v2_cifar10.h5')
        except:
            st.error("Error: Could not find the model file 'resnet50v2_cifar10.h5'")
            st.stop()
    
    def preprocess_image(self, img):
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to 32x32 (CIFAR-10 size)
        img = cv2.resize(img, (32, 32))
        
        # Normalize pixel values
        img = img.astype('float32') / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def predict(self, image):
        processed_image = self.preprocess_image(image)
        prediction = self.model.predict(processed_image, verbose=0)
        class_idx = np.argmax(prediction[0])
        confidence = np.max(prediction[0]) * 100
        return self.class_names[class_idx], confidence

def main():
    st.title("Real-Time Object Classification")
    st.write("Using ResNet50V2 trained on CIFAR-10 dataset")
    
    # Initialize the classifier
    classifier = ObjectClassifier()
    
    # Add a sidebar
    st.sidebar.title("Controls")
    prediction_mode = st.sidebar.radio(
        "Choose Prediction Mode",
        ("Webcam", "Upload Image")
    )
    
    if prediction_mode == "Webcam":
        # Start webcam
        st.write("## Webcam Feed")
        run = st.checkbox('Start Webcam')
        FRAME_WINDOW = st.image([])
        
        if run:
            cap = cv2.VideoCapture(0)
            while run:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to access webcam")
                    break
                
                # Make prediction
                class_name, confidence = classifier.predict(frame)
                
                # Draw prediction on frame
                height, width = frame.shape[:2]
                text = f"{class_name}: {confidence:.1f}%"
                cv2.putText(frame, text, (10, height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame)
            
            cap.release()
    
    else:  # Upload Image mode
        st.write("## Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Convert uploaded file to opencv image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Display original image
            st.write("### Original Image")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Make prediction
            class_name, confidence = classifier.predict(image)
            
            # Display results
            st.write("### Prediction Results")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Class", class_name)
            with col2:
                st.metric("Confidence", f"{confidence:.1f}%")

if __name__ == "__main__":
    main()