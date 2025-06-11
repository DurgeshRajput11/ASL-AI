import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import mediapipe as mp
import tempfile
import os
import json
import time
from typing import List, Dict, Optional
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="ASL Recognition App",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #262730;  /* dark gray-blue */
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ---- Load your model ONCE for all users ----
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("finetuned_model.h5")

MODEL = load_model()

class ASLStreamlitApp:
    def __init__(self):
        self.asl_classes = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            'SPACE', 'DELETE', 'NOTHING'
        ]
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = []
        if 'current_word' not in st.session_state:
            st.session_state.current_word = ""

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        if image.shape[:2] != (224, 224):
            image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)
        return image

    def extract_hand_region(self, image: np.ndarray) -> Optional[np.ndarray]:
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_image)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    h, w, _ = image.shape
                    x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
                    y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
                    x_min, x_max = int(min(x_coords)), int(max(x_coords))
                    y_min, y_max = int(min(y_coords)), int(max(y_coords))
                    padding = 40
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(w, x_max + padding)
                    y_max = min(h, y_max + padding)
                    hand_region = image[y_min:y_max, x_min:x_max]
                    if hand_region.size > 0:
                        return hand_region, (x_min, y_min, x_max, y_max)
            return None, None
        except Exception as e:
            st.error(f"Error extracting hand: {str(e)}")
            return None, None

    def predict_sign(self, image: np.ndarray, use_hand_detection: bool = True) -> Dict:
        if MODEL is None:
            st.error("Model not loaded!")
            return {}
        try:
            original_image = image.copy()
            hand_detected = False
            bbox = None
            if use_hand_detection:
                hand_region, bbox = self.extract_hand_region(image)
                if hand_region is not None:
                    image = hand_region
                    hand_detected = True
                else:
                    st.warning("No hand detected, using full image")
            processed_image = self.preprocess_image(image)
            predictions = MODEL.predict(processed_image, verbose=0)
            top_indices = np.argsort(predictions[0])[::-1][:5]
            results = {
                'predictions': predictions[0],
                'predicted_class': self.asl_classes[top_indices[0]],
                'confidence': float(predictions[0][top_indices[0]]),
                'top_predictions': [
                    {
                        'class': self.asl_classes[idx],
                        'confidence': float(predictions[0][idx])
                    }
                    for idx in top_indices
                ],
                'hand_detected': hand_detected,
                'bbox': bbox,
                'original_image': original_image,
                'processed_image': image
            }
            return results
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return {}

    def display_prediction_results(self, results: Dict):
        if not results:
            return
        predicted_class = results['predicted_class']
        confidence = results['confidence']
        if confidence > 0.8:
            conf_class = "confidence-high"
        elif confidence > 0.5:
            conf_class = "confidence-medium"
        else:
            conf_class = "confidence-low"
        st.markdown(f"""
        <div class="prediction-box">
            <h2>🎯 Prediction: {predicted_class}</h2>
            <p class="{conf_class}">Confidence: {confidence:.2%}</p>
            <p>Hand Detected: {'✅ Yes' if results['hand_detected'] else '❌ No'}</p>
        </div>
        """, unsafe_allow_html=True)
        top_preds = results['top_predictions']
        df_preds = pd.DataFrame(top_preds)
        fig = px.bar(
            df_preds,
            x='confidence',
            y='class',
            orientation='h',
            title="Top 5 Predictions",
            color='confidence',
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.prediction_history.append({
            'timestamp': timestamp,
            'prediction': predicted_class,
            'confidence': confidence
        })

    def display_image_with_detection(self, results: Dict):
        if not results or 'original_image' not in results:
            return
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            original = results['original_image']
            if results['hand_detected'] and results['bbox']:
                x_min, y_min, x_max, y_max = results['bbox']
                cv2.rectangle(original, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
                cv2.putText(original, "Hand Detected", (x_min, y_min-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            st.image(original, channels="BGR", use_column_width=True)
        with col2:
            st.subheader("Processed Region")
            processed = results['processed_image']
            st.image(processed, channels="BGR", use_column_width=True)

    def word_builder_interface(self):
        st.subheader("🔤 Word Builder")
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            current_word = st.text_input(
                "Current Word:",
                value=st.session_state.current_word,
                key="word_display"
            )
            st.session_state.current_word = current_word
        with col2:
            if st.button("Clear Word"):
                st.session_state.current_word = ""
                st.experimental_rerun()
        with col3:
            if st.button("Save Word"):
                if st.session_state.current_word:
                    st.success(f"Saved: '{st.session_state.current_word}'")
                    # Save to file/db if needed

    def prediction_history_interface(self):
        st.subheader("📊 Prediction History")
        if st.session_state.prediction_history:
            df_history = pd.DataFrame(st.session_state.prediction_history)
            st.write("Recent Predictions:")
            st.dataframe(df_history.tail(10), use_container_width=True)
            if len(df_history) > 1:
                pred_counts = df_history['prediction'].value_counts().head(10)
                fig = px.pie(
                    values=pred_counts.values,
                    names=pred_counts.index,
                    title="Prediction Frequency"
                )
                st.plotly_chart(fig, use_container_width=True)
            if st.button("Clear History"):
                st.session_state.prediction_history = []
                st.experimental_rerun()
        else:
            st.info("No predictions yet. Upload an image to get started!")

    def run(self):
        st.markdown('<h1 class="main-header">🤟 ASL Alphabet Recognition</h1>',
                   unsafe_allow_html=True)
        with st.sidebar:
            st.header("⚙️ Settings")
            st.subheader("Detection Settings")
            use_hand_detection = st.checkbox("Use Hand Detection", value=True)
            confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
            st.subheader("ℹ️ About")
            st.info("""
            This app recognizes American Sign Language alphabet signs.
            **Features:**
            - Real-time hand detection
            - High-accuracy CNN models
            - Word building interface
            - Prediction history
            **Classes:** A-Z, SPACE, DELETE, NOTHING
            """)

        tab1, tab2, tab3, tab4 = st.tabs(["📷 Image Recognition", "🎥 Video Processing", "🔤 Word Builder", "📊 History"])
        with tab1:
            st.header("Image Recognition")
            uploaded_file = st.file_uploader(
                "Upload an image",
                type=['png', 'jpg', 'jpeg'],
                help="Upload an image containing an ASL alphabet sign"
            )
            camera_image = st.camera_input("Or take a photo")
            image_to_process = uploaded_file or camera_image
            if image_to_process is not None:
                image = Image.open(image_to_process)
                image_array = np.array(image)
                if len(image_array.shape) == 3:
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                if MODEL is not None:
                    with st.spinner("Making prediction..."):
                        results = self.predict_sign(image_array, use_hand_detection)
                    if results:
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            self.display_prediction_results(results)
                        with col2:
                            self.display_image_with_detection(results)
                        if results['confidence'] > confidence_threshold:
                            predicted_class = results['predicted_class']
                            if st.button(f"Add '{predicted_class}' to word"):
                                if predicted_class == "SPACE":
                                    st.session_state.current_word += " "
                                elif predicted_class == "DELETE":
                                    if st.session_state.current_word:
                                        st.session_state.current_word = st.session_state.current_word[:-1]
                                elif predicted_class != "NOTHING":
                                    st.session_state.current_word += predicted_class
                                st.experimental_rerun()
                else:
                    st.warning("Model not loaded!")
        with tab2:
            st.header("Video Processing")
            st.info("Video processing feature - Upload a video file for frame-by-frame ASL recognition")
            video_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
            if video_file is not None:
                st.video(video_file)
                if st.button("Process Video"):
                    st.info("Video processing functionality would go here")
        with tab3:
            self.word_builder_interface()
        with tab4:
            self.prediction_history_interface()
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666;'>
            Made with ❤️ using Streamlit | ASL Recognition System
        </div>
        """, unsafe_allow_html=True)

def main():
    app = ASLStreamlitApp()
    app.run()

if __name__ == "__main__":
    main()
