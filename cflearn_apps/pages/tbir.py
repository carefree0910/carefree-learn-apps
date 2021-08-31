import os
import json
import cflearn_deploy

import streamlit as st

from typing import Any
from sqlmodel import select
from sqlmodel import Session
from requests import Response
from cflearn_deploy.toolkit import bytes_to_np
from cflearn_deploy.api_utils import post_json


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
def get_indices_response(text: str, **kwargs: Any) -> Response:
    with st.spinner("Fetching retrieval indices..."):
        return post_json([text], uri="/cv/tbir", **kwargs)


def app() -> None:
    st.title("Text Based Image Retrieval")
    engine = cflearn_deploy.get_engine(file=sqlite_file, echo=False)

    metric_type = st.sidebar.radio("metric_type", ["L2", "IP"], index=0)
    top_k = st.sidebar.slider("Top K", min_value=3, max_value=48, value=9)
    num_probe = st.sidebar.slider("num probe", min_value=8, max_value=24, value=16)
    model = st.sidebar.text_input("Model Name", "tbir")

    text = st.text_input("Please input your text!", "A poster of noodle")
    if text:
        columns = st.columns(3)
        indices_response = get_indices_response(
            text,
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
