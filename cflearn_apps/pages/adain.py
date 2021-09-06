import numpy as np
import streamlit as st

from PIL import Image
from typing import Any
from requests import Response
from cflearn_deploy.toolkit import bytes_to_np
from cflearn_deploy.api_utils import post_img_arr


@st.cache
def get_response(content: np.ndarray, style: np.ndarray, **kwargs: Any) -> Response:
    with st.spinner("Generating stylized image..."):
        return post_img_arr(content, style, uri="/cv/adain", **kwargs)


def app() -> None:
    st.title("Arbitrary Style Transfer with AdaIN")
    model = st.sidebar.text_input("Model Name", "adain")

    content_file = st.file_uploader("Please upload the content file")
    style_file = st.file_uploader("Please upload the style file")
    contrast_q = st.sidebar.slider("contrast_q", min_value=0.0, max_value=0.2, value=0.01)
    col1, col2, col3 = st.columns(3)
    content = style = None
    if content_file is not None:
        content = Image.open(content_file).convert("RGB")
        content.thumbnail((256, 256), Image.ANTIALIAS)
        col1.image(content, caption="Content Image")
    if style_file is not None:
        style = Image.open(style_file).convert("RGB")
        style.thumbnail((256, 256), Image.ANTIALIAS)
        col2.image(style, caption="Style Image")
    if content is not None and style is not None:
        content_arr, style_arr = map(np.array, [content, style])
        response = get_response(content_arr, style_arr, model_name=model, q=contrast_q)
        if not response.ok:
            st.markdown(f"**Failed to get stylized image! ({response.reason})**")
        else:
            stylized = bytes_to_np(response.content, mode="RGB")
            col3.image(stylized, caption="Stylized")
