import numpy as np
import streamlit as st

from PIL import Image
from typing import Any
from cflearn_deploy.toolkit import resize_to
from cflearn_deploy.client import Client


@st.cache
def get_stylized(content: np.ndarray, style: np.ndarray, **kwargs: Any) -> np.ndarray:
    with st.spinner("Generating stylized image..."):
        inputs = {"input": content[None, ...], "style": style[None, ...]}
        results = Client().infer("adain", inputs, **kwargs)
        return results.as_numpy("predictions")[0]


def app() -> None:
    content_file = st.file_uploader("Please upload the content file")
    style_file = st.file_uploader("Please upload the style file")
    col1, col2, col3 = st.columns(3)
    content = style = original_shape = None
    if content_file is not None:
        content = Image.open(content_file).convert("RGB")
        original_shape = content.size
        col1.image(content, caption="Content Image")
        content.thumbnail((512, 512), Image.ANTIALIAS)
    if style_file is not None:
        style = Image.open(style_file).convert("RGB")
        col2.image(style, caption="Style Image")
        style.thumbnail((512, 512), Image.ANTIALIAS)
    if content is not None and style is not None and original_shape is not None:
        content_arr, style_arr = map(np.array, [content, style])
        stylized = get_stylized(content_arr, style_arr)
        resized = resize_to(stylized.astype(np.float32) / 255.0, original_shape)
        col3.image(resized, caption="Stylized")
