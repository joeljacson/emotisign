import streamlit as st
import cv2
from deepface import DeepFace
from PIL import Image
import numpy as np
import pyttsx3
import tempfile
import time

# Title
st.title("🧠 EmotiSign - Web Emotion Analyzer")
st.write("Analyze your emotion from an image, get voice + emoji feedback.")

# Text-to-Speech
engine = pyttsx3.init()

# Emoji-based response
emotion_responses = {
    "happy": "😊 You look happy! Enjoy your day!",
    "sad": "😢 You seem sad. Sending you positive vibes!",
    "angry": "😠 You look angry. Try to relax.",
    "surprise": "😮 You look surprised! What happened?",
    "fear": "😨 You look a bit scared. Take it easy.",
    "neutral": "😐 You look calm and neutral.",
    "disgust": "😣 You look displeased. Everything okay?"
}

# Upload or take webcam photo
option = st.radio("Choose Input Method", ["📤 Upload Image", "📸 Take Webcam Photo"])

img = None

if option == "📤 Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

elif option == "📸 Take Webcam Photo":
    img_bytes = st.camera_input("Take a photo")
    if img_bytes:
        img = Image.open(img_bytes)
        st.image(img, caption="Webcam Image", use_column_width=True)

# Emotion Detection
if img:
    with st.spinner("🔍 Analyzing emotion..."):
        try:
            img_np = np.array(img)
            result = DeepFace.analyze(img_np, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']

            response = emotion_responses.get(emotion, f"You look {emotion}")
            st.success(f"**Emotion Detected:** {emotion.capitalize()}")
            st.markdown(f"### {response}")

            # Speak response
            if st.button("🔊 Hear Feedback"):
                engine.say(response)
                engine.runAndWait()

        except Exception as e:
            st.error(f"Error: {e}")
