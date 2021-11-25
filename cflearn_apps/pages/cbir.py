import numpy as np
import streamlit as st

from PIL import Image
from skimage.transform import resize

from .utils import info_retrieval
from .utils import image_retrieval


def app() -> None:
    top_k = st.sidebar.slider("Top K", min_value=3, max_value=48, value=9)
    num_probe = st.sidebar.slider("num probe", min_value=8, max_value=24, value=16)
    model = st.sidebar.text_input("Model Name", "cbir")
    task = st.sidebar.radio(
        "Task",
        [
            "poster",
            "caption",
        ],
        index=0,
    )
    if task == "poster":
        img_type = "RGB"
        src_folder = "poster"
        gray = model.endswith("_shape")
    elif task == "caption":
        img_type = "RGB"
        src_folder = "cbir.caption"
        gray = None
    else:
        raise ValueError

    uploaded_file = st.file_uploader("Please upload your file")
    if uploaded_file is not None:
        col1, *columns = st.columns(2 if task == "caption" else 4)
        with st.spinner("Uploading image..."):
            image = Image.open(uploaded_file).convert(img_type)
            image.thumbnail((256, 256), Image.ANTIALIAS)
            col1.image(image, caption="Uploaded Image")
            img_arr = np.array(image)
            resized_img = resize(img_arr, (224, 224), mode="constant")
            resized_img = resized_img.astype(np.float32)
            if task == "poster":
                image_retrieval(
                    resized_img[None, ...],
                    src_folder,
                    "cbir",
                    top_k,
                    num_probe,
                    columns,
                    gray,
                )
            elif task == "caption":
                for info in info_retrieval(
                    resized_img[None, ...],
                    src_folder,
                    "cbir.caption",
                    top_k,
                    num_probe,
                    gray,
                ):
                    columns[0].selectbox(str(info.distance), info.info, 0)
