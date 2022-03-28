import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import os
import io
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


def mask_video():
    # global result_img
    cap = cv2.VideoCapture('./images/out_video.mp4')
    ret, img = cap.read()
    facenet = cv2.dnn.readNet('./models/deploy.prototxt', './models/res10_300x300_ssd_iter_140000.caffemodel')
    model = load_model('./models/mask_detector.model')

    fourcc = cv2.VideoWriter_fourcc('X', '2', '6', '4')  # need openh264
    out = cv2.VideoWriter('./images/out_video_result.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (img.shape[1], img.shape[0]))

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        h, w = img.shape[:2]

        blob = cv2.dnn.blobFromImage(img, scalefactor=1., size=(300, 300), mean=(104., 177., 123.))
        facenet.setInput(blob)
        dets = facenet.forward()

        result_img = img.copy()

        for i in range(dets.shape[2]):
            confidence = dets[0, 0, i, 2]
            if confidence < 0.5:
                continue

            x1 = int(dets[0, 0, i, 3] * w)
            y1 = int(dets[0, 0, i, 4] * h)
            x2 = int(dets[0, 0, i, 5] * w)
            y2 = int(dets[0, 0, i, 6] * h)

            face = img[y1:y2, x1:x2]

            face_input = cv2.resize(face, dsize=(224, 224))
            face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)
            face_input = preprocess_input(face_input)
            face_input = np.expand_dims(face_input, axis=0)

            mask, nomask = model.predict(face_input).squeeze()

            if mask > nomask:
                color = (0, 255, 0)
                label = 'Mask %d%%' % (mask * 100)
            else:
                color = (0, 0, 255)
                label = 'No Mask %d%%' % (nomask * 100)

            cv2.rectangle(result_img, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=color, lineType=cv2.LINE_AA)
            cv2.putText(result_img, text=label, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                        color=color, thickness=2, lineType=cv2.LINE_AA)

        out.write(result_img)
    out.release()
    cap.release()



def app():
    st.subheader("Detection on Video")
    uploaded_file = st.file_uploader("Upload Video", type=["mp4"])
    temporary_location = False

    if uploaded_file is not None:
        our_video = io.BytesIO(uploaded_file.read())  ## BytesIO Object
        st.video(our_video, start_time=0)
        temporary_location = "./images/out_video.mp4"

        with open(temporary_location, 'wb') as out1:  ## Open temporary file as bytes
            out1.write(our_video.read())  ## Read bytes into file

        # close file
        # out.close()
        if st.button('Process'):
            mask_video()
            vid_file = open("./images/out_video_result.mp4", "rb").read()
            st.video(vid_file, start_time=0)



