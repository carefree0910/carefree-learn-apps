import numpy as np
import streamlit as st

from PIL import Image
from requests import Response
from skimage.transform import resize
from cflearn_deploy.toolkit import cutout
from cflearn_deploy.toolkit import to_uint8
from cflearn_deploy.toolkit import bytes_to_np
from cflearn_deploy.api_utils import post_img_arr


@st.cache
def get_rgba_response(src: np.ndarray, smooth: int, tight: float) -> Response:
    with st.spinner("Generating RGBA image..."):
        return post_img_arr(src, uri="/cv/sod", smooth=smooth, tight=tight)


def app() -> None:
    st.title("Salient Object Detection")

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
        image.thumbnail((320, 320), Image.ANTIALIAS)
        col1.image(image, caption="Uploaded Image")
        img_arr = np.array(image)
        normalized_img = img_arr.astype(np.float32) / 255.0
        resized_img = resize(img_arr, (320, 320), mode="constant")
        resized_img = resized_img.astype(np.float32)
        rgba_response = get_rgba_response(resized_img, smooth=smooth, tight=tight)
        if not rgba_response.ok:
            st.markdown(f"**Failed to get alpha mask! ({rgba_response.reason})**")
        else:
            alpha = bytes_to_np(rgba_response.content, mode="RGBA")[..., -1]
            if thresh is not None:
                alpha = (alpha > thresh).astype(np.float32)
            alpha, rgba = cutout(normalized_img, alpha, smooth, tight)
            col2.image(to_uint8(alpha), caption="Mask")
            st.image(rgba, caption="RGBA")
