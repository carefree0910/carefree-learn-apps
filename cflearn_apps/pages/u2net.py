import os
import cflearn

import numpy as np
import streamlit as st

from PIL import Image
from cflearn import cv
from cflearn.misc.toolkit import to_uint8
from cflearn.api.cv.models.u2net import cutout

from .utils import download_with_progress
from ..constants import MODEL_FOLDER


supported_models = {
    "large_cascade_lite_512_bce_iou": "20210804.0",
    "lite_finetune": "20210804.0",
}


def get_url(model: str) -> str:
    prefix = "https://github.com/carefree0910/carefree-learn-models/releases/download"
    return f"{prefix}/{supported_models[model]}/{model}.onnx"


@st.cache
def get_alpha(api: cv.U2NetAPIWithONNX, src: np.ndarray) -> np.ndarray:
    with st.spinner("Generating alpha mask..."):
        return api._get_alpha(src)


def app() -> None:
    st.title("Cutout Demo")

    models = sorted(supported_models)
    model = st.sidebar.radio("Select a model", models)
    use_threshold = st.sidebar.radio("Use threshold", [True, False])
    thresh = None
    if use_threshold:
        thresh = st.sidebar.slider("Threshold", min_value=0.0, max_value=1.0, value=0.5)
    smooth = st.sidebar.slider("Smooth", min_value=0, max_value=20, value=4)
    tight = st.sidebar.slider("Tight", min_value=0.0, max_value=1.0, value=0.9)

    onnx_path = os.path.join(MODEL_FOLDER, f"{model}.onnx")
    if not os.path.isfile(onnx_path):
        url = get_url(model)
        download_with_progress(url, onnx_path)

    with st.spinner(f"Loading onnx model ({model})..."):
        api = cflearn.cv.U2NetAPIWithONNX(onnx_path)
    st.markdown(f"**onnx model `{model}` loaded!**")
    uploaded_file = st.file_uploader("Please upload your file")
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image")
        arr = np.array(image).astype(np.float32) / 255.0
        alpha = get_alpha(api, arr)
        alpha, rgba = cutout(arr, alpha, smooth, tight, None)
        if thresh is not None:
            alpha = (alpha > thresh).astype(np.float32)
        st.image(to_uint8(alpha), caption="Mask")
        st.image(rgba, caption="RGBA")
