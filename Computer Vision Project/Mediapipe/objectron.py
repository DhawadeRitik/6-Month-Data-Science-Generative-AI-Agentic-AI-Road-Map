import cv2
import mediapipe as mp
import streamlit as st
import numpy as np

# Title
st.title("Objectron Model with Mediapipe")

# Initialize Mediapipe
mp_objectron = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils

# Choose object
objects = st.selectbox("Select the object", options=['Chair', 'Cup', 'Camera', 'Shoe'])

# Choose input type
input_type = st.selectbox("Choose input type", options=['Image', 'Webcam', 'Upload Video'])

# Setup Objectron
objectron = mp_objectron.Objectron(
    static_image_mode=(input_type == 'Image'),
    max_num_objects=5,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_name=objects
)

# --- IMAGE detection ---
if input_type == 'Image':
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        # Convert BGR → RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = objectron.process(rgb_frame)

        if results.detected_objects:
            for detected_object in results.detected_objects:
                mp_drawing.draw_landmarks(frame, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
                mp_drawing.draw_axis(frame, detected_object.rotation, detected_object.translation)

        st.image(frame, channels="BGR")

# --- WEBCAM detection ---
elif input_type == 'Webcam':
    cam_source = st.radio("Choose camera:", ["Internal Webcam (0)", "External Webcam (1)"])
    cam_index = 0 if cam_source == "Internal Webcam (0)" else 1

    video_display = st.image([])
    start = st.checkbox("Start Webcam Detection")

    if start:
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            st.error("Unable to access webcam")
        else:
            while start:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read frame from webcam")
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = objectron.process(rgb_frame)

                if results.detected_objects:
                    for detected_object in results.detected_objects:
                        mp_drawing.draw_landmarks(frame, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
                        mp_drawing.draw_axis(frame, detected_object.rotation, detected_object.translation)

                video_display.image(frame, channels="BGR")

        cap.release()

# --- UPLOAD VIDEO detection ---
elif input_type == 'Upload Video':
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = f"temp_video.mp4"
        with open(tfile, "wb") as f:
            f.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile)
        video_display = st.image([])

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = objectron.process(rgb_frame)

            if results.detected_objects:
                for detected_object in results.detected_objects:
                    mp_drawing.draw_landmarks(frame, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
                    mp_drawing.draw_axis(frame, detected_object.rotation, detected_object.translation)

            video_display.image(frame, channels="BGR")

        cap.release()
