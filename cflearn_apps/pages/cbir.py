import numpy as np
import streamlit as st

from PIL import Image
from skimage.transform import resize
from cflearn_deploy.api_utils import post_img_arr

from .utils import image_retrieval


def fonts_caption_callback(file: str) -> str:
    return file.split("/")[-2]


def app() -> None:
    top_k = st.sidebar.slider("Top K", min_value=3, max_value=48, value=9)
    num_probe = st.sidebar.slider("num probe", min_value=8, max_value=24, value=16)
    model = st.sidebar.text_input("Model Name", "cbir")
    task = st.sidebar.radio(
        "Task",
        [
            "poster",
            "fonts",
        ],
        index=0,
    )
    model_name = f"{model}.{task}"
    if task == "poster":
        img_type = "RGB"
        src_folder = "poster"
        gray = False
        no_transform = False
        caption_callback = None
    elif task == "fonts":
        img_type = "L"
        src_folder = "fonts/english"
        gray = True
        no_transform = True
        caption_callback = fonts_caption_callback
    else:
        raise ValueError

    uploaded_file = st.file_uploader("Please upload your file")
    if uploaded_file is not None:
        col1, *columns = st.columns(4)
        with st.spinner("Uploading image..."):
            image = Image.open(uploaded_file).convert(img_type)
            image.thumbnail((256, 256), Image.ANTIALIAS)
            col1.image(image, caption="Uploaded Image")
            img_arr = np.array(image)
            resized_img = resize(img_arr, (224, 224), mode="constant")
            resized_img = resized_img.astype(np.float32)
            image_retrieval(
                "cbir",
                post_img_arr,
                resized_img,
                task,
                src_folder,
                model_name,
                top_k,
                num_probe,
                columns,
                gray,
                no_transform,
                caption_callback,
            )
