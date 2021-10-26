import numpy as np
import streamlit as st

from PIL import Image
from typing import Any
from skimage.transform import resize
from cflearn_deploy.toolkit import cutout
from cflearn_deploy.toolkit import to_uint8
from cflearn_deploy.toolkit import resize_to
from cflearn_deploy.client import Client


@st.cache
def get_alpha(src: np.ndarray, **kwargs: Any) -> np.ndarray:
    with st.spinner("Generating RGBA image..."):
        results = Client().infer("sod", {"input": src[None, ...]}, **kwargs)
        return results.as_numpy("predictions")[0]


def app() -> None:
    use_threshold = st.sidebar.radio("Use threshold", [True, False], index=1)
    thresh = None
    if use_threshold:
        thresh = st.sidebar.slider("Threshold", min_value=0.0, max_value=1.0, value=0.5)
    smooth = st.sidebar.slider("Smooth", min_value=0, max_value=20, value=0)
    tight = st.sidebar.slider("Tight", min_value=0.0, max_value=1.0, value=0.9)

    uploaded_file = st.file_uploader("Please upload your file")
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        image = Image.open(uploaded_file).convert("RGB")
        original_shape = image.size
        original_normalized_img = np.array(image).astype(np.float32) / 255.0
        col1.image(image, caption="Uploaded Image")
        image.thumbnail((320, 320), Image.LANCZOS)
        resized_img = resize(np.array(image), (320, 320), mode="constant")
        alpha = get_alpha(to_uint8(resized_img))
        if thresh is not None:
            alpha = (alpha > thresh).astype(np.float32)
        alpha = np.array(resize_to(alpha, original_shape))
        alpha = alpha.astype(np.float32) / 255.0
        alpha, rgba = cutout(original_normalized_img, alpha, smooth, tight)
        col2.image(to_uint8(alpha), caption="Mask")
        st.image(rgba, caption="RGBA")
