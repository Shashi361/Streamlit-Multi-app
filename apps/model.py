import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def app():
    st.write(
        '''
    This project workflow follow 3 phases
    1. Loading the Face Mask Classifier model
    2. Detect Faces in the image
    3. Extract each Face Region of Interest(ROI)
    4. Apply face mask classifier to each face ROI to determine 'mask' or 'No mask'
''')