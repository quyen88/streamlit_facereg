import streamlit as st 
import pickle 
import yaml 
import pandas as pd
import numpy as np
import cv2
from PIL import Image
cfg = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
PKL_PATH = cfg['PATH']["PKL_PATH"]
st.set_page_config(layout="wide")

#Load databse 
with open(PKL_PATH, 'rb') as file:
    database = pickle.load(file)
# Index, Id, Name, Image  = st.columns([0.5,0.5,3,3])
df = pd.DataFrame.from_dict(database, orient='index')
num_columns = 3
image_height = 200
# Lặp qua DataFrame và hiển thị thông tin và hình ảnh
for index, row in df.iterrows():
    if index % num_columns == 0:
        cols = st.columns(num_columns)  # Tạo một hàng mới
    with cols[index % num_columns]:
        st.write(f"Name: {row['name']} (ID: {row['id']})")
        img = Image.fromarray(np.uint8(row['image']))
        # Cắt hoặc thay đổi kích thước hình ảnh để có chiều cao cố định
        # img = img.resize((int(img.width * (image_height / img.height)), image_height))
        # Hiển thị hình ảnh
        st.image(img, width=200)
        


  
# for idx, person in database.items():
#     with Index:
#         st.write(idx)
#     with Id:
#         st.write(person['id'])
#     with Name: 
#         st.write(person['name'])
#     with Image: 
#         st.image(person['image'],width=200)

