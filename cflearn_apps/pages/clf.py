import json

import numpy as np
import streamlit as st

from PIL import Image
from typing import Any
from requests import Response
from skimage.transform import resize
from cflearn_deploy.api_utils import post_img_arr


@st.cache
def get_prob_response(src: np.ndarray, **kwargs: Any) -> Response:
    with st.spinner("Generating probabilities..."):
        return post_img_arr(src, uri="/cv/clf", **kwargs)


def app() -> None:
    top_k = st.sidebar.slider("Smooth", min_value=1, max_value=20, value=5)
    model = st.sidebar.text_input("Model Name", "cct")

    uploaded_file = st.file_uploader("Please upload your file")
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image.thumbnail((224, 224), Image.ANTIALIAS)
        col1, col2 = st.columns(2)
        col1.image(image, caption="Uploaded Image")
        img_arr = np.array(image)
        resized_img = resize(img_arr, (224, 224), mode="constant")
        resized_img = resized_img.astype(np.float32)
        prob_response = get_prob_response(resized_img, model_name=model)
        if not prob_response.ok:
            st.markdown(f"**Failed to get probabilities! ({prob_response.reason})**")
        else:
            rs = json.loads(prob_response.content)
            probabilities = np.array(rs["probabilities"], dtype=np.float32).tolist()
            top_indices = np.argsort(probabilities).tolist()[::-1][:top_k]
            col2.json({str(i): probabilities[i] for i in top_indices})
