import cv2
import mediapipe as mp
import streamlit as st
import tempfile
import os
import numpy as np

# Title
st.title("FaceMesh Detection")

# Sidebar controls
with st.sidebar:
    mode = st.radio("Choose Mode", ["Webcam", "Image", "Video"])
    
    connection_option = st.selectbox(
        "Select the option for connection",
        options=["FACEMESH_TESSELATION", "FACEMESH_CONTOURS", "FACEMESH_IRISES"]
    )

    # RGB sliders
    color1 = st.slider("Red", 0, 255, 0)
    color2 = st.slider("Green", 0, 255, 255)
    color3 = st.slider("Blue", 0, 255, 0)

    # Line thickness & point size
    thickness = st.slider("Line Thickness", 1, 5, 1)
    circle_radius = st.slider("Point Size", 1, 3, 1)

# Map dropdown to MediaPipe constants
mp_facemesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
connection_dict = {
    "FACEMESH_TESSELATION": mp_facemesh.FACEMESH_TESSELATION,
    "FACEMESH_CONTOURS": mp_facemesh.FACEMESH_CONTOURS,
    "FACEMESH_IRISES": mp_facemesh.FACEMESH_IRISES
}
selected_connection = connection_dict[connection_option]

drawing_spec = mp_drawing.DrawingSpec(
    color=(color1, color2, color3), thickness=thickness, circle_radius=circle_radius
)

# Function to process a frame
def process_frame(frame, face_mesh):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=selected_connection,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_spec
            )
    return frame

# Initialize FaceMesh
face_mesh = mp_facemesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Mode 1: Webcam
if mode == "Webcam":
    start = st.checkbox("Start Webcam")
    if start:
        cap = cv2.VideoCapture(0)
        col1, col2 = st.columns(2)
        frame_placeholder = col1.empty()
        processed_placeholder = col2.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Unable to access webcam")
                break

            processed = process_frame(frame.copy(), face_mesh)

            frame_placeholder.image(frame, channels="BGR", caption="Original Webcam")
            processed_placeholder.image(processed, channels="BGR", caption="FaceMesh Processed")

        cap.release()

# Mode 2: Image Upload
elif mode == "Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        col1, col2 = st.columns(2)
        file_bytes = uploaded_image.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        processed = process_frame(img.copy(), face_mesh)

        col1.image(img, channels="BGR", caption="Original Image")
        col2.image(processed, channels="BGR", caption="FaceMesh Processed")

# Mode 3: Video Upload
elif mode == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        col1, col2 = st.columns(2)
        frame_placeholder = col1.empty()
        processed_placeholder = col2.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed = process_frame(frame.copy(), face_mesh)

            frame_placeholder.image(frame, channels="BGR", caption="Original Video")
            processed_placeholder.image(processed, channels="BGR", caption="FaceMesh Processed")

        cap.release()
        os.remove(tfile.name)
