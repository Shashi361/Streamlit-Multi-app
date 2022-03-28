import streamlit as st
from multiapp import MultiApp
from apps import home, uploadimage, model, web, video  # import your app modules here

app = MultiApp()

#st.set_page_config(page_title='Face Mask Detector', page_icon='😷', layout='centered')
st.markdown('<h1 align="center">😷 Face Mask Detection</h1>', unsafe_allow_html=True)
st.markdown("""Built with OpenCV and Keras/TensorFlow leveraging Deep Learning and Computer Vision Concepts to detect 
face mask in still images as well as in real-time webcam streaming 
""")

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Image ", uploadimage.app)
app.add_app("About", model.app)
app.add_app("webcam", web.app)
app.add_app("video", video.app)
# The main app
app.run()
