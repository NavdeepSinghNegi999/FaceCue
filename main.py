import keras
import streamlit as st
import numpy as np
import cv2
from helper import *
import webbrowser
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode


# @st.cache(allow_output_mutation=True)
# def load_my_model():
#     model = keras.models.load_model("VGG16_model.h5")
#     return model
# model = load_my_model()
model = keras.models.load_model(r"model/VGG16_model.h5")




def index_to_emotion(index):
    emotion_labels = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}
    index_to_emotion = {index:emotion for emotion, index in emotion_labels.items()}
    emotion = index_to_emotion[index]
    return emotion


def model_prediction(test_image):
    
    image = keras.preprocessing.image.load_img(test_image, target_size=(150,150))
    img_to_arr = keras.preprocessing.image.img_to_array(image)
    
    img_arr = np.array([img_to_arr]) 
    img_arr = img_arr/255.0
    
    prediction = model.predict(img_arr)
    result_index = np.argmax(prediction)
    
    return result_index

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")



#sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Image Emotion Prediction"])

# Home Page
if(app_mode == "Image Emotion Prediction"):
    st.header("Emotion Detection")
    
    # Load image
    test_image = st.file_uploader("Choose an Image")
    
    if(st.button("Show Image")):
        st.image(test_image, use_column_width=True)
    
    if(st.button("Predict")):
        st.write("Our Prediction")
        result_index  = model_prediction(test_image)
        result = index_to_emotion(result_index)
        st.success(f"{result}")

        if(st.button("Recommend")):
            st.write("Our Prediction")
    
            result = "angry"
            st.success(f"{result}")
            size = len(emotion_video_recommendations[result])
            index = int(np.random.rand()*size)
            video_type = emotion_video_recommendations[result][index]
            st.components.v1.html(f"<iframe width='560' height='315' src='https://www.youtube.com/results?search_query={video_type}' frameborder='0'></iframe>")

            # webbrowser.open(f"https://www.youtube.com/results?search_query={'happy'}")
        
