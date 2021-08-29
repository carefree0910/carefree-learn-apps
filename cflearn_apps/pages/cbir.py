import os
import json
import cflearn_deploy

import numpy as np
import streamlit as st

from PIL import Image
from typing import Any
from sqlmodel import select
from sqlmodel import Session
from requests import Response
from skimage.transform import resize
from cflearn_deploy.toolkit import bytes_to_np
from cflearn_deploy.api_utils import post_img_arr


root = os.path.dirname(__file__)
sqlite_file = os.path.join(
    root,
    os.pardir,
    os.pardir,
    os.pardir,
    "carefree-learn-deploy",
    "apis",
    "data",
    "sqlite.db",
)


@st.cache
def get_indices_response(src: np.ndarray, **kwargs: Any) -> Response:
    with st.spinner("Fetching retrieval indices..."):
        return post_img_arr(src, uri="/cv/cbir", **kwargs)


def app() -> None:
    st.title("Content Based Image Retrieval")
    engine = cflearn_deploy.get_engine(file=sqlite_file, echo=False)

    metric_type = st.sidebar.radio("metric_type", ["L2", "IP"], index=0)
    top_k = st.sidebar.slider("Top K", min_value=3, max_value=48, value=9)
    num_probe = st.sidebar.slider("num probe", min_value=8, max_value=24, value=16)
    model = st.sidebar.text_input("Model Name", "dino_vit")

    uploaded_file = st.file_uploader("Please upload your file")
    if uploaded_file is not None:
        col1, *columns = st.columns(4)
        with st.spinner("Uploading image..."):
            image = Image.open(uploaded_file).convert("RGB")
            image.thumbnail((256, 256), Image.ANTIALIAS)
            col1.image(image, caption="Uploaded Image")
            img_arr = np.array(image)
            resized_img = resize(img_arr, (224, 224), mode="constant")
            resized_img = resized_img.astype(np.float32)
        indices_response = get_indices_response(
            resized_img,
            model_name=model,
            top_k=top_k,
            nprobe=num_probe,
            metric_type=metric_type,
        )
        if not indices_response.ok:
            reason = indices_response.reason
            st.markdown(f"**Failed to get retrieval indices! ({reason})**")
        else:
            rs = json.loads(indices_response.content)
            indices, distances = rs["indices"], rs["distances"]
            item_base = cflearn_deploy.ImageItem
            with Session(engine) as session:
                for i, (idx, distance) in enumerate(zip(indices, distances)):
                    rs = session.exec(select(item_base.bytes).where(item_base.id == idx))
                    np_img = bytes_to_np(rs.one(), mode="RGB")
                    columns[i % 3].image(np_img, caption=f"{distance:8.6f}")
