import os
import json
import time
import urllib.error
import urllib.request

import streamlit as st

from PIL import Image
from stqdm import stqdm
from typing import Any
from typing import Callable
from typing import Optional
from requests import Response
from functools import wraps

from ..constants import IMAGES_FOLDER


def retry(exception: Any, tries: int = 4, delay: int = 3, backoff: int = 2):
    def deco_retry(f):
        @wraps(f)
        def f_retry(*args, **kwargs):
            tries_, delay_ = tries, delay
            while tries_ > 1:
                try:
                    return f(*args, **kwargs)
                except exception as e:
                    print(f"{e}, Retrying in {delay_} seconds...")
                    time.sleep(delay_)
                    tries_ -= 1
                    delay_ *= backoff
            return f(*args, **kwargs)

        return f_retry

    return deco_retry


@retry((urllib.error.HTTPError, ConnectionResetError))
def download_with_progress(url: str, tgt_path: str) -> None:
    folder = os.path.dirname(tgt_path)
    with DownloadProgressBar(
        unit="B",
        unit_scale=True,
        miniters=1,
        desc=url.split("/")[-1],
    ) as t:
        urllib.request.urlretrieve(url, tgt_path, reporthook=t.update_to)


class DownloadProgressBar(stqdm):
    def update_to(self, b: int, bsize: int, total: int):
        self.total = total
        self.update(min(self.total, b * bsize - self.n))

    def st_display(self, n: Optional[int], total: Optional[int], **kwargs: Any) -> None:
        if n is not None and total is not None:
            n = min(n, total)
        super().st_display(n, total, **kwargs)


@st.cache
def get_indices_response(
    uri: str,
    post_fn: Callable,
    query: Any,
    **kwargs: Any,
) -> Response:
    with st.spinner("Fetching retrieval indices..."):
        return post_fn(query, uri=f"/cv/{uri}", **kwargs)


def image_retrieval(
    uri: str,
    post_fn: Callable,
    query: Any,
    task: str,
    model: str,
    top_k: int,
    num_probe: int,
    columns: tuple,
) -> None:
    indices_response = get_indices_response(
        uri,
        post_fn,
        query,
        task=task,
        onnx_name=model,
        top_k=top_k,
        nprobe=num_probe,
    )
    if not indices_response.ok:
        reason = indices_response.reason
        st.markdown(f"**Failed to get retrieval indices! ({reason})**")
    else:
        rs = json.loads(indices_response.content)
        files, distances = rs["files"], rs["distances"]
        task_img_folder = os.path.join(IMAGES_FOLDER, task)
        for i, (file, distance) in enumerate(zip(files, distances)):
            img = Image.open(os.path.join(task_img_folder, file))
            columns[i % 3].image(img, caption=f"{distance:8.6f}")
