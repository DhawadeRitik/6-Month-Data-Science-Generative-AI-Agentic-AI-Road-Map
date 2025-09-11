import cv2 
import mediapipe as mp 
import streamlit as st
import time

# Title of the app
st.title("Hand Landmark Detection Using Mediapipe")

# Initialize mediapipe hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Capture the video
video = cv2.VideoCapture(0)

# Create a window to display the video
video_display = st.image([])

# Checkbox to start/stop video
start_video = st.checkbox("Start Video", help="Unselect checkbox to stop video")

# Add FPS display
fps_placeholder = st.empty()

if not video.isOpened():
    st.error("Unable to capture the video")
    
else:
    prev_time = 0
    
    while start_video:
        
        ret, frame = video.read()
        
        if not ret:
            st.error("Unable to read the frame")
            break

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        result = hands.process(rgb_frame)

        # Draw landmarks
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(rgb_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                
        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time

        # Show FPS on frame (optional: draw it on image using OpenCV)
        cv2.putText(rgb_frame, f'FPS: {int(fps)}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame in Streamlit
        video_display.image(rgb_frame)

    video.release()
