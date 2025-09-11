import cv2
import numpy as np
import streamlit as st
import tempfile


# Title
st.title("🎥 Real-Time Motion Detection using Background Subtraction")

# Upload video
video_uploader = st.file_uploader("Upload a video here", type=["mp4", "mov", "avi"])

# Checkbox to control playback
start_stop_video = st.checkbox("Start Video", help="Unselect to stop")

if video_uploader is not None:
    
    # save the uploaded video to the temporarly file 
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(video_uploader.read())

    cap = cv2.VideoCapture(temp_file.name)

    if not cap.isOpened():
        st.error("❌ Video could not be opened.")
    else:
        # Create background subtractor
        back_sub = cv2.createBackgroundSubtractorMOG2()

        # Two columns for display
        col1, col2 = st.columns(2)
        original_placeholder = col1.empty()
        motion_placeholder = col2.empty()

        while start_stop_video and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ End of video reached or cannot read frame.")
                break

            # Resize for faster processing (optional)
            frame = cv2.resize(frame, (500, 340))

            # Convert BGR → RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Motion detection mask
            motion_mask = back_sub.apply(frame)
            motion_rgb = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2RGB)

            # Display in Streamlit
            original_placeholder.image(frame_rgb, channels="RGB", caption="Original Video")
            motion_placeholder.image(motion_rgb, channels="RGB", caption="Motion Detection")

        cap.release()

