
import streamlit as st
import cv2
import face_recognition as frg
import yaml 
from utils import recognize, build_dataset, recognizeVideo
import tempfile
import random
import time


st.set_page_config(layout="wide", page_title="FaceRecognition")
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

cfg = yaml.load(open('config.yaml','r'),Loader=yaml.FullLoader)
PICTURE_PROMPT = cfg['INFO']['PICTURE_PROMPT']
WEBCAM_PROMPT = cfg['INFO']['WEBCAM_PROMPT']



st.sidebar.title("Settings")



#Create a menu bar
menu = ["Picture","Webcam", "Video"]
choice = st.sidebar.selectbox("Input type",menu)
#Put slide to adjust tolerance
TOLERANCE = st.sidebar.slider("Tolerance",0.0,1.0,0.35,0.01)
st.sidebar.info("Ngưỡng nhận dạng. Khoảng cách giữa các khuôn mặt, nhỏ sự tương đồng càng lớn")

#Infomation section 
# st.sidebar.title("Thông tin Nhận dạng")
# name_container = st.sidebar.empty()
# id_container = st.sidebar.empty()
# name_container.info('Name: Unnknown')
# id_container.success('ID: Unknown')
if choice == "Picture":
    st.title("Face Recognition")
    st.info("Nhận dạng Ảnh")
    uploaded_images = st.file_uploader("Upload",type=['jpg','png','jpeg'],accept_multiple_files=True)
    if len(uploaded_images) != 0:
        #Read uploaded image with face_recognition
        for image in uploaded_images:
            image = frg.load_image_file(image)
            image, name, id = recognize(image,TOLERANCE) 
            # name_container.info(f"Name: {name}")
            # id_container.success(f"ID: {id}")
            st.image(image)
    # else: 
    #     st.info("Please upload an image")
    
elif choice == "Webcam":
    st.title("Face Recognition")
    st.info("Nhận dạng Webcam")
    #Camera Settings
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    FRAME_WINDOW = st.image([])
    
    while True:
        ret, frame = cam.read()
        if not ret:
            st.error("Không có kết nối Webcam")
            st.info("Vui lòng kết nối Webcam")
            st.stop()
        
        image, name, id = recognize(frame,TOLERANCE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(image)
        # if name != "Unknown" and id != "Unknown":
        #     st.image(image)
        
        
elif choice == "Video":
    st.title("Face Recognition")
    st.info("Nhận dạng từ Video")
    uploaded_video = st.file_uploader("Tải lên video", type=["mp4", "avi"])
    FRAME_WINDOW = st.image([])

    if uploaded_video is not None:
        temp_video = tempfile.NamedTemporaryFile(delete=False)
        temp_video.write(uploaded_video.read())
        video_stream = cv2.VideoCapture(temp_video.name)
        if not video_stream.isOpened():
            st.error("Không thể mở video.")
        else:
            frame_count = 0
            # processing_rate = 5  # Xử lý 5 frame mỗi giây
            # start_time = time.time()
            
            process_this_frame = True  
            while True:
                ret, frame = video_stream.read()
                if not ret:
                    st.info("Kết thúc video.")
                    break
                # frame_count += 1
                # if frame_count % 40 == 0:
                # if frame_count >= processing_rate * (time.time() - start_time):
                if process_this_frame:
                    small_frame = cv2.resize(frame, (0, 0), fx=1/3, fy=1/3)
                    image, name, id = recognize(small_frame,TOLERANCE)
                    small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    FRAME_WINDOW.image(small_frame)
                    # if name != "Unknown" and id != "Unknown":
                    #     print(name)
                    #     st.image(small_frame)
                process_this_frame = not process_this_frame
                    # frame_count = 0
                    # start_time = time.time()
           
        temp_video.close()




with st.sidebar.form(key='my_form'):
    st.title("For Dev")
    submit_button = st.form_submit_button(label='REBUILD DATASET')
    if submit_button:
        with st.spinner("Rebuilding dataset..."):
            build_dataset()
        st.success("Dataset has been reset")