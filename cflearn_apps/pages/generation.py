import numpy as np
import streamlit as st

from typing import Any
from cflearn_deploy.client import Client


def get_generated(model: str, **kwargs: Any) -> np.ndarray:
    with st.spinner("Generating image..."):
        results = Client().infer("sg2", {"model": model}, **kwargs)
        return results.as_numpy("predictions")


def app() -> None:
    model = st.sidebar.radio(
        "Model Type",
        [
            "cat",
            "dog",
            "wild",
            "ffhq",
            "metfaces",
        ],
        index=0,
    )
    mapping = {
        "cat": "cat",
        "dog": "dog",
        "wild": "animal",
        "ffhq": "person",
        "metfaces": "art piece",
    }
    generated = get_generated(model)
    col1, col2 = st.columns(2)
    col1.subheader(f"This {mapping[model]} does not exist!")
    col2.button("Refresh!")
    st.image(generated)
