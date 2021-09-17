import streamlit as st

from typing import Any
from requests import Response
from cflearn_deploy.toolkit import bytes_to_np
from cflearn_deploy.api_utils import post_img_arr


def get_rgb_response(**kwargs: Any) -> Response:
    with st.spinner("Generating image..."):
        return post_img_arr(uri="/cv/style_gan", **kwargs)


def app() -> None:
    model = st.sidebar.radio(
        "Model Name",
        [
            "afhqcat",
            "afhqdog",
            "afhqwild",
            "ffhq",
            "metfaces",
        ],
        index=0,
    )
    mapping = {
        "afhqcat": "cat",
        "afhqdog": "dog",
        "afhqwild": "animal",
        "ffhq": "person",
        "metfaces": "art piece",
    }
    rgb_response = get_rgb_response(onnx_name=model)
    if not rgb_response.ok:
        st.markdown(f"**Failed to generate image! ({rgb_response.reason})**")
    else:
        col1, col2 = st.columns(2)
        col1.subheader(f"This {mapping[model]} does not exist!")
        col2.button("Refresh!")
        st.image(bytes_to_np(rgb_response.content, mode="RGB"))
