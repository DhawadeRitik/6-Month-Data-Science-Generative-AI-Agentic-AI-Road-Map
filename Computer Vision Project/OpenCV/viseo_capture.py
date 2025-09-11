import cv2
import numpy 
import streamlit as st 

# title of the app 
st.title('OpenCV Video Capture with Streamlit')

st.markdown('Please select the checkbox to capture video & unselect to stop')
# start /stop 
run = st.checkbox('Start Webcam')

# frame to display the video
frame_window = st.image([])


# capture the video 

video = cv2.VideoCapture(0)

while run :
    # read the video frame
    _ , frame = video.read()
    
    if not _:
        st.error('Fail to capture video')
        break
    
    # convert the image BGR to RGB 
    frame = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
    
    # show the image in streamlit 
    frame_window.image(frame)
    
    
# release webcam after stoping the webcam feed
video.release()