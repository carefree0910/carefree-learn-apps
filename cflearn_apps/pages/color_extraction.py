import json
import math

import numpy as np
import streamlit as st

from PIL import Image
from PIL import ImageDraw
from typing import Any
from requests import Response
from cflearn_deploy.api_utils import post_img_arr


@st.cache
def get_colors_response(src: np.ndarray, **kwargs: Any) -> Response:
    with st.spinner("Extracting colors..."):
        return post_img_arr(src, uri="/cv/color_extraction", **kwargs)


def app() -> None:
    num_colors = st.sidebar.slider("Num Colors", min_value=1, max_value=20, value=5)
    size = st.sidebar.slider("Display Size", min_value=50, max_value=150, value=100)
    per_row = st.sidebar.slider("Num Per Row", min_value=2, max_value=8, value=4)

    uploaded_file = st.file_uploader("Please upload your file")
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)
        col1.image(image, caption="Uploaded Image")
        image.thumbnail((256, 256), Image.ANTIALIAS)
        colors_response = get_colors_response(np.array(image), num_colors=num_colors)
        if not colors_response.ok:
            st.markdown(f"**Failed to extract colors! ({colors_response.reason})**")
        else:
            canvas_w = size * min(num_colors, per_row)
            canvas_h = size * math.ceil(num_colors / per_row)
            canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
            draw = ImageDraw.Draw(canvas)
            colors = json.loads(colors_response.content)["colors"]
            for i, color in enumerate(colors):
                lx = (i % per_row) * size
                ly = (i // per_row) * size
                draw.rectangle(
                    (lx, ly, lx + size, ly + size),
                    fill=tuple(color),
                    outline=(220, 220, 220),
                    width=2,
                )
            col2.image(canvas)
