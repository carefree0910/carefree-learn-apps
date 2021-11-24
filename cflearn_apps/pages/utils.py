import os
import json

import numpy as np
import streamlit as st

from PIL import Image
from PIL import ImageFile
from typing import Any
from typing import Dict
from typing import List
from typing import Callable
from typing import Optional
from typing import NamedTuple
from cflearn_deploy.client import Client

from ..constants import META_FOLDER
from ..constants import IMAGES_FOLDER

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


@st.cache
def get_info(
    model: str,
    query: np.ndarray,
    top_k: int,
    num_probe: int,
    gray: Optional[bool] = None,
    **kwargs: Any,
) -> Dict[str, List[Any]]:
    with st.spinner("Fetching retrieval indices..."):
        inputs = {"input": query, "top_k": top_k, "num_probe": num_probe}
        if gray is not None:
            inputs["gray"] = gray
        results = Client().infer(model, inputs, **kwargs)
        return {
            "files": [f.decode() for f in results.as_numpy("files")[0].tolist()],
            "distances": results.as_numpy("distances")[0].tolist(),
        }


def image_retrieval(
    query: np.ndarray,
    src_folder: str,
    model: str,
    top_k: int,
    num_probe: int,
    columns: tuple,
    gray: Optional[bool] = None,
    caption_callback: Optional[Callable[[str], str]] = None,
) -> None:
    info = get_info(model, query, top_k, num_probe, gray)
    files, distances = info["files"], info["distances"]
    img_folder = os.path.join(IMAGES_FOLDER, src_folder)
    for i, (file, distance) in enumerate(zip(files, distances)):
        img = Image.open(os.path.join(img_folder, file)).convert("RGB")
        caption = f"{distance:8.6f}"
        if caption_callback is not None:
            caption = f"{caption}, {caption_callback(file)}"
        columns[i % 3].image(img, caption=caption)


class Info(NamedTuple):
    info: Any
    distance: float


def info_retrieval(
    query: np.ndarray,
    src_folder: str,
    model: str,
    top_k: int,
    num_probe: int,
    gray: Optional[bool] = None,
) -> List[Info]:
    info = get_info(model, query, top_k, num_probe, gray)
    files, distances = info["files"], info["distances"]
    src_folder = os.path.join(META_FOLDER, src_folder)
    results = []
    for i, (file, distance) in enumerate(zip(files, distances)):
        name = os.path.splitext(file)[0]
        with open(os.path.join(src_folder, f"{name}.json"), "r", encoding="utf-8") as f:
            results.append(Info(json.load(f), distance))
    return results
