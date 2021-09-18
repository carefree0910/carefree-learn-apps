import numpy as np
import streamlit as st

from PIL import Image
from typing import Any
from requests import Response
from cflearn_deploy.toolkit import resize_to
from cflearn_deploy.toolkit import bytes_to_np
from cflearn_deploy.toolkit import quantile_normalize
from cflearn_deploy.api_utils import post_img_arr


@st.cache
def get_response(model: str, content: np.ndarray, **kwargs: Any) -> Response:
    with st.spinner("Generating stylized image..."):
        return post_img_arr(content, uri=f"/cv/{model}", **kwargs)


def app() -> None:
    model = st.sidebar.text_input("Model Name", "cycle_gan")
    sub_type = st.sidebar.radio(
        "Model Sub Type",
        [
            "pixel_art",
        ],
        index=0,
    )
    model_name = f"{model}.{sub_type}"

    content_file = st.file_uploader("Please upload the content file")
    col1, col2 = st.columns(2)
    if content_file is not None:
        content = Image.open(content_file).convert("RGB")
        original_shape = content.size
        col1.image(content, caption="Content Image")
        content.thumbnail((512, 512), Image.ANTIALIAS)
        response = get_response(model, np.array(content), onnx_name=model_name)
        if not response.ok:
            st.markdown(f"**Failed to get stylized image! ({response.reason})**")
        else:
            stylized = bytes_to_np(response.content, mode="RGB")
            stylized = quantile_normalize(stylized)
            col2.image(resize_to(stylized, original_shape), caption="Stylized")
