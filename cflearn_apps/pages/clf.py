import os
import json

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from PIL import Image
from typing import Any
from requests import Response
from skimage.transform import resize
from cflearn_deploy.api_utils import post_img_arr
from cflearn_deploy.constants import LABEL_KEY
from cflearn_deploy.constants import PREDICTIONS_KEY

from ..constants import META_FOLDER


@st.cache
def get_prob_response(src: np.ndarray, **kwargs: Any) -> Response:
    with st.spinner("Generating probabilities..."):
        return post_img_arr(src, uri="/cv/clf", **kwargs)


def app() -> None:
    top_k = st.sidebar.slider("Top K", min_value=1, max_value=20, value=5)
    model = st.sidebar.text_input("Model Name", "clf")
    task = st.sidebar.radio(
        "Task",
        [
            "products",
            "fonts",
        ],
        index=0,
    )
    model_name = f"{model}.{task}"

    uploaded_file = st.file_uploader("Please upload your file")
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)
        col1.image(image, caption="Uploaded Image")
        image.thumbnail((224, 224), Image.ANTIALIAS)
        resized_img = resize(np.array(image), (224, 224), mode="constant")
        resized_img = resized_img.astype(np.float32)
        kwargs = {}
        if task == "fonts":
            kwargs["gray"] = True
            kwargs["no_transform"] = True
        prob_response = get_prob_response(resized_img, onnx_name=model_name, **kwargs)
        if not prob_response.ok:
            st.markdown(f"**Failed to get probabilities! ({prob_response.reason})**")
        else:
            probabilities = json.loads(prob_response.content)["probabilities"]
            meta_folder = os.path.join(META_FOLDER, task)
            for k, v in probabilities.items():
                # fetch
                if k == PREDICTIONS_KEY:
                    k = "main"
                mapping_file = f"idx2{k if k != 'main' else f'{LABEL_KEY}'}.json"
                mapping_path = os.path.join(meta_folder, mapping_file)
                if not os.path.isfile(mapping_path):
                    mapping = None
                else:
                    with open(mapping_path, "r") as f:
                        mapping = json.load(f)
                top_indices = np.argsort(v).tolist()[::-1][:top_k]
                top_probabilities = [v[i] for i in top_indices]
                top_probabilities.append(1.0 - sum(top_probabilities) - 1.0e-8)
                # draw
                kwargs = dict(figsize=(12, 7), subplot_kw=dict(aspect="equal"), dpi=80)
                fig, ax = plt.subplots(**kwargs)
                wedges, texts, auto_texts = ax.pie(
                    top_probabilities,
                    autopct=lambda value: f"{value:.1f}%",
                    textprops=dict(color="w"),
                    colors=plt.cm.Dark2.colors,
                    startangle=140,
                    normalize=False,
                )
                categories = [str(i) for i in top_indices]
                if mapping is not None:
                    categories = list(map(mapping.get, categories))
                categories.append("else")
                ax.legend(
                    wedges,
                    categories,
                    title="Class",
                    loc="center left",
                    bbox_to_anchor=(1, 0, 0.5, 1),
                )
                plt.setp(auto_texts, size=10, weight=700)
                ax.set_title(f"Pie Chart for '{k}'")
                col2.pyplot(fig)
