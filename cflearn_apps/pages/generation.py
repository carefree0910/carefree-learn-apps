import streamlit as st

from typing import Any
from requests import Response
from cflearn_deploy.toolkit import bytes_to_np
from cflearn_deploy.api_utils import post_img_arr


def get_rgb_response(**kwargs: Any) -> Response:
    with st.spinner("Generating image..."):
        return post_img_arr(uri="/cv/style_gan", **kwargs)


def app() -> None:
    model = st.sidebar.text_input("Model Name", "style_gan")
    sub_type = st.sidebar.radio(
        "Model Sub Type",
        [
            "afhqcat",
            "afhqdog",
            "afhqwild",
            "ffhq",
            "metfaces",
        ],
        index=0,
    )
    model_name = f"{model}.{sub_type}"
    mapping = {
        "style_gan.afhqcat": "cat",
        "style_gan.afhqdog": "dog",
        "style_gan.afhqwild": "animal",
        "style_gan.ffhq": "person",
        "style_gan.metfaces": "art piece",
    }
    rgb_response = get_rgb_response(onnx_name=model_name)
    if not rgb_response.ok:
        st.markdown(f"**Failed to generate image! ({rgb_response.reason})**")
    else:
        col1, col2 = st.columns(2)
        col1.subheader(f"This {mapping[model_name]} does not exist!")
        col2.button("Refresh!")
        st.image(bytes_to_np(rgb_response.content, mode="RGB"))
