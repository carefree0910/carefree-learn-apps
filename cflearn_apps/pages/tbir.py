import os
import json
import random
import hashlib
import cflearn_deploy
import http.client
import urllib.parse

import streamlit as st

from typing import Any
from sqlmodel import select
from sqlmodel import Session
from requests import Response
from cflearn_deploy.toolkit import bytes_to_np
from cflearn_deploy.api_utils import post_json


root = os.path.dirname(__file__)
info_folder = os.path.join(root, "info")
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


def zh2en(text: str, app_id: str, secret_key: str) -> str:
    salt = random.randint(32768, 65536)
    sign = f"{app_id}{text}{salt}{secret_key}"
    sign = hashlib.md5(sign.encode()).hexdigest()
    client = http.client.HTTPConnection('api.fanyi.baidu.com')
    client.request(
        "GET",
        f"/api/trans/vip/translate?appid={app_id}&q={urllib.parse.quote(text)}"
        f"&from=zh&to=en&salt={salt}&sign={sign}",
    )
    response = client.getresponse()
    result_all = response.read().decode("utf-8")
    result = json.loads(result_all)
    return result["trans_result"][0]["dst"]


def app() -> None:
    st.title("Text Based Image Retrieval")
    engine = cflearn_deploy.get_engine(file=sqlite_file, echo=False)

    metric_type = st.sidebar.radio("metric_type", ["L2", "IP"], index=0)
    top_k = st.sidebar.slider("Top K", min_value=3, max_value=48, value=9)
    num_probe = st.sidebar.slider("num probe", min_value=8, max_value=24, value=16)
    model = st.sidebar.text_input("Model Name", "tbir")

    with open(os.path.join(info_folder, "baidu_fanyi.json"), "r") as f:
        fanyi = json.load(f)
        app_id, secret_key = fanyi["app_id"], fanyi["secret_key"]

    text = st.text_input("Please input your text!", "来点卡通风格的")
    if text:
        text = zh2en(text, app_id, secret_key)
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
