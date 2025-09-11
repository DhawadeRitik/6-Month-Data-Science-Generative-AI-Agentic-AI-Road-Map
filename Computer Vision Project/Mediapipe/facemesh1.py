import cv2
import mediapipe as mp
import streamlit as st

# Title of the page
st.title("FaceMesh Detection")

# Checkbox to start/stop
start = st.checkbox("Start FaceMesh Detection")

# Placeholder for video frame
video_display = st.empty()

# Initialize mediapipe facemesh
mp_facemesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Capture video
video = cv2.VideoCapture(0)

with st.sidebar :
    
    mode = st.radio("Select the Mode of Video Capture", options=['Internal Webcam', 'Image', 'Video'])
    
    connection_option = st.selectbox("Select the oprtion for connection", options=['FACEMESH_TESSELATION','FACEMESH_CONTOURS','FACEMESH_IRISES'])
    
    color1 = st.slider("Select value for Red", min_value=0, max_value=255, value=0)
    color2 = st.slider("Select value for Green", min_value=0, max_value=255, value=255)
    color3 = st.slider("Select value for Blue", min_value=0, max_value=255, value=0)
    
    thickness = st.slider("FaceMesh thickness", min_value = 1, max_value=5)
    circle_radius = st.slider("FaceMesh Circle Radius", min_value = 1, max_value=5)
    
    
    
connection_dict = {
    "FACEMESH_TESSELATION": mp_facemesh.FACEMESH_TESSELATION,
    "FACEMESH_CONTOURS": mp_facemesh.FACEMESH_CONTOURS,
    "FACEMESH_IRISES": mp_facemesh.FACEMESH_IRISES
}

selected_connection = connection_dict[connection_option]

max_faces = st.slider("Number of Face", min_value = 1, max_value = 5)

with mp_facemesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=max_faces,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    if not video.isOpened():
        st.error("Unable to capture video")
    else:
        while start:
            success, frame = video.read()
            if not success:
                st.error("Unable to capture frame")
                break

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            result = face_mesh.process(rgb_frame)
            rgb_frame.flags.writeable = True

            # Back to BGR for OpenCV display
            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

            # Draw landmarks if faces detected
            if result.multi_face_landmarks:
                for face_landmarks in result.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=bgr_frame,
                        landmark_list=face_landmarks,
                        connections=selected_connection,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(
                            color=(color1, color2, color3), thickness=thickness, circle_radius=circle_radius
                        )
                    )

            # Show frame in Streamlit
            video_display.image(bgr_frame, channels="BGR")

        video.release()
