import os
import io
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

load_dotenv()
try:
    VISION_KEY = os.environ['VISION_KEY']
    VISION_ENDPOINT = os.environ['VISION_ENDPOINT']
except:
    print('Missing required env variables')
    exit()

try:
    client = ImageAnalysisClient(
        endpoint=VISION_ENDPOINT,
        credential=AzureKeyCredential(VISION_KEY)
    )
except:
    st.error('Cannot connect to Azure Computer Vision')
    exit()


st.title('Computer Vision with Streamlit')

uploadedFile = st.file_uploader('Choose image', type=['jpg', 'jpeg'])

if uploadedFile:
    image = Image.open(uploadedFile)
    st.image(image, caption='Uploaded image')

    imageBytes = io.BytesIO()
    image.save(imageBytes, format=image.format)
    imageBytes = imageBytes.getvalue()

    if st.button('Analyze image'):
        try:
            visual_features = [
                VisualFeatures.TAGS,
                VisualFeatures.CAPTION,
                VisualFeatures.DENSE_CAPTIONS
            ]

            result = client.analyze(
                image_data=imageBytes,
                visual_features=visual_features
            )

            if result.caption:
                st.write("Caption:")
                st.write(f'{result.caption.text}')
                st.write(f'{result.caption.confidence:.4f}')

            if len(result.dense_captions.list) > 0:
                st.write('Dense Captions')
                st.dataframe(result.dense_captions.list)

            if len(result.tags.list) > 0:
                st.write('Tags')    
                st.dataframe(result.tags.list)
        except Exception as e:
            st.error(f'There was an error when analysing the image {e}')