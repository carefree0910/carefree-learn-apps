import os
import json

import streamlit as st

from PIL import Image
from PIL import ImageFile
from typing import Any
from typing import Callable
from typing import Optional
from requests import Response

from ..constants import IMAGES_FOLDER

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


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
    src_folder: str,
    model: str,
    top_k: int,
    num_probe: int,
    columns: tuple,
    gray: bool = False,
    no_transform: bool = False,
    caption_callback: Optional[Callable[[str], str]] = None,
) -> None:
    indices_response = get_indices_response(
        uri,
        post_fn,
        query,
        task=task,
        onnx_name=model,
        top_k=top_k,
        nprobe=num_probe,
        gray=gray,
        no_transform=no_transform,
    )
    if not indices_response.ok:
        reason = indices_response.reason
        st.markdown(f"**Failed to get retrieval indices! ({reason})**")
    else:
        rs = json.loads(indices_response.content)
        files, distances = rs["files"], rs["distances"]
        img_folder = os.path.join(IMAGES_FOLDER, src_folder)
        for k, k_files in files.items():
            if len(files) > 1:
                columns[0].markdown("---")
                columns[1].markdown(
                    f"<h3 style='text-align: center'> {k} </h3>",
                    unsafe_allow_html=True,
                )
                columns[2].markdown("---")
            k_distances = distances[k]
            for i, (file, distance) in enumerate(zip(k_files, k_distances)):
                img = Image.open(os.path.join(img_folder, file)).convert("RGB")
                caption = f"{distance:8.6f}"
                if caption_callback is not None:
                    caption = f"{caption}, {caption_callback(file)}"
                columns[i % 3].image(img, caption=caption)
