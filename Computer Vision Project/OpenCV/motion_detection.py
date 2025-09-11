import cv2 
import numpy 
import streamlit as st 


video = cv2.VideoCapture(0)

video_display_window = st.image([])

if not video.isOpened():
    st.error('E=Unable to capture video')
    exit()
  
start_video = st.checkbox("Start Video Capture")
back_sub = cv2.createBackgroundSubtractorMOG2()
    
while start_video :
    
    ret, frame = video.read()
    
    if not ret:
        st.error('Unable to read the frame')
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    motion_mask = back_sub.apply(frame)
    motion_rgb = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2RGB)
    
    # video_display_window.image(rgb_frame)
    video_display_window.image(motion_rgb)
    
video.release()