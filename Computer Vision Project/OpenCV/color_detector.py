import cv2
import numpy as np
import streamlit as st 

# Title
st.title("Color Detector Using OpenCV")

# Sidebar options
with st.sidebar:
    st.header("Color Detection Options")
    color_detection_option = st.selectbox(
        "Select the color you want to detect", 
        options=['red', 'green', 'blue', 'yellow', 'orange', 'pink', 'voilet', 'purple', 'cyan','white'],
        index=None
    )

# Checkbox to run webcam
run = st.checkbox("Start Web Cam")

# Open video
video = cv2.VideoCapture(0)

# Define HSV ranges
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


# Two columns layout created once
col1, col2 = st.columns(2)
with col1:
    st.subheader("Original Video")
    original_placeholder = st.image([])
with col2:
    subheader_placeholder = st.empty()  # placeholder for title (updates with color)
    detected_placeholder = st.image([])

# Loop
while run:
    ret, frame = video.read()
    if not ret:
        st.error("Could not capture webcam frame")
        break 

    # Convert for display
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Original Image 
    original_placeholder.image(rgb_image)
    
    if color_detection_option:
        # Read current color selection dynamically
        lower, upper = color_ranges[color_detection_option]
        lower_range = np.array(lower)
        upper_range = np.array(upper)

        # Mask and detection
        mask = cv2.inRange(hsv_image, lower_range, upper_range)
        detected = cv2.bitwise_and(frame, frame, mask=mask)
        detected = cv2.cvtColor(detected, cv2.COLOR_BGR2RGB)

        # Update placeholders dynamically
        subheader_placeholder.subheader(f"{color_detection_option.capitalize()} Color Detection")
        detected_placeholder.image(detected)

video.release()
