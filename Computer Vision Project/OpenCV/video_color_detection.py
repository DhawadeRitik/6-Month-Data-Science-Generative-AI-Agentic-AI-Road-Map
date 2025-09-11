import streamlit as st
import cv2
import numpy as np
import time

st.title("Color Detection")

start_video = st.checkbox("Select to capture video")

video_placeholder = st.image([])


with st.sidebar:
    st.header("Color Detection Options")
    color_detection_option = st.selectbox(
        "Select the color you want to detect", 
        options=['red', 'green', 'blue', 'yellow', 'orange', 'pink', 'voilet', 'purple', 'cyan','white']
        
    )

color_ranges = {
    'red': ([161, 155, 84], [179, 255, 255]),
    'green': ([40, 40, 40], [70, 255, 255]),
    'blue': ([94, 80, 2], [126, 255, 255]),
    'yellow': ([20, 100, 100], [30, 255, 255]),
    'orange': ([5, 150, 150], [15, 255, 255]),
    'pink': ([140, 50, 50], [170, 255, 255]),
    'violet': ([130, 50, 50], [160, 255, 255]),
    'purple': ([125, 50, 50], [150, 255, 255]),
    'cyan': ([85, 50, 50], [95, 255, 255]),
    'white': ([0, 0, 200], [180, 40, 255])
}



if start_video:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Webcam not working...")
    else:
        try:
            while start_video:
                ret, frame = cap.read()
                if not ret:
                    st.error("Frame not captured")
                    break

                # Convert BGR to HSV
                hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                # select the colors to detect 
                lower, upper = color_ranges[color_detection_option]
                # convert it to numpy array color range
                lower_color = np.array(lower)
                upper_color = np.array(upper)
                mask = cv2.inRange(hsv_frame, lower_color, upper_color)
                red_detected = cv2.bitwise_and(frame, frame, mask=mask)

                # Convert to RGB for Streamlit
                red_detected_rgb = cv2.cvtColor(red_detected, cv2.COLOR_BGR2RGB)
                video_placeholder.image(red_detected_rgb)

                # Small delay to allow UI update
                time.sleep(0.03)
        finally:
            cap.release()
