import os
import json
import random
import hashlib
import http.client
import urllib.parse

import streamlit as st

from cflearn_deploy.api_utils import post_json

from .utils import image_retrieval
from ..constants import API_INFO_FOLDER


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
    task = st.sidebar.text_input("Task", "poster")
    model = st.sidebar.text_input("Model Name", "tbir")
    top_k = st.sidebar.slider("Top K", min_value=3, max_value=48, value=9)
    num_probe = st.sidebar.slider("num probe", min_value=8, max_value=24, value=16)

    with open(os.path.join(API_INFO_FOLDER, "baidu_fanyi.json"), "r") as f:
        fanyi = json.load(f)
        app_id, secret_key = fanyi["app_id"], fanyi["secret_key"]

    text = st.text_input("Please input your text!", "来点卡通风格的")
    if text:
        text = zh2en(text, app_id, secret_key)
        image_retrieval(
            "tbir",
            post_json,
            [text],
            task,
            task,
            model,
            top_k,
            num_probe,
            st.columns(3),
        )
