import cv2
import mediapipe as mp
import streamlit as st
import numpy as np

# Title
st.title("MediaPipe FaceMesh with Streamlit")

# Mediapipe initialization
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# User controls
max_faces = st.slider("Max number of faces", 1, 5, 1)
refine_landmarks = st.checkbox("Refine landmarks (for irises)", value=True)
input_type = st.selectbox("Input Type", ["Webcam", "Image"])

# Placeholder for displaying video or image
display_frame = st.image([])

# --- IMAGE DETECTION ---
if input_type == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process FaceMesh
        with mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:
            results = face_mesh.process(frame_rgb)

        # Draw landmarks
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )
                if refine_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                    )
        display_frame.image(frame, channels="BGR")

# --- WEBCAM DETECTION ---
elif input_type == "Webcam":
    cam_source = st.radio("Choose camera:", ["Internal Webcam (0)", "External Webcam (1)"])
    cam_index = 0 if cam_source == "Internal Webcam (0)" else 1

    start = st.checkbox("Start Webcam")
    if start:
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            st.error("Unable to access webcam")
        else:
            with mp_face_mesh.FaceMesh(
                max_num_faces=max_faces,
                refine_landmarks=refine_landmarks,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            ) as face_mesh:
                while start:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Failed to read frame from webcam")
                        break

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(frame_rgb)

                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            mp_drawing.draw_landmarks(
                                image=frame,
                                landmark_list=face_landmarks,
                                connections=mp_face_mesh.FACEMESH_TESSELATION,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                            )
                            mp_drawing.draw_landmarks(
                                image=frame,
                                landmark_list=face_landmarks,
                                connections=mp_face_mesh.FACEMESH_CONTOURS,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                            )
                            if refine_landmarks:
                                mp_drawing.draw_landmarks(
                                    image=frame,
                                    landmark_list=face_landmarks,
                                    connections=mp_face_mesh.FACEMESH_IRISES,
                                    landmark_drawing_spec=None,
                                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                                )
                    # Mirror image
                    frame = cv2.flip(frame, 1)
                    display_frame.image(frame, channels="BGR")

            cap.release()
