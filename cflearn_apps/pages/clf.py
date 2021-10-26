import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from PIL import Image
from typing import Any
from typing import Dict
from skimage.transform import resize
from cflearn_deploy.client import Client
from cflearn_deploy.constants import PREDICTIONS_KEY

plt.rcParams["font.sans-serif"] = ["Noto Sans SC"]
output_names = {
    "product": ["predictions", "classes"],
}


@st.cache
def get_info(
    src: np.ndarray,
    top_k: int,
    task: str,
    **kwargs: Any,
) -> Dict[str, Dict[str, Any]]:
    with st.spinner("Generating probabilities..."):
        inputs = {"input": src[None, ...], "top_k": top_k}
        results = Client().infer(f"{task}_clf", inputs, **kwargs)
        info = {}
        for name in output_names[task]:
            response = results.as_numpy(name)[0].tolist()
            splits = [cls.decode().strip().split(":") for cls in response]
            info[name] = {
                "classes": [":".join(split[2:]) for split in splits],
                "probabilities": [float(split[0]) for split in splits],
            }
        return info


def app() -> None:
    top_k = st.sidebar.slider("Top K", min_value=1, max_value=20, value=5)
    task = st.sidebar.radio(
        "Task",
        [
            "product",
        ],
        index=0,
    )

    uploaded_file = st.file_uploader("Please upload your file")
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)
        col1.image(image, caption="Uploaded Image")
        image.thumbnail((224, 224), Image.ANTIALIAS)
        resized_img = resize(np.array(image), (224, 224), mode="constant")
        resized_img = resized_img.astype(np.float32)
        info = get_info(resized_img, top_k, task)
        for k, k_info in info.items():
            # fetch
            if k == PREDICTIONS_KEY:
                k = "main"
            eps = 1.0e-8
            top_probabilities = [max(eps, v) for v in k_info["probabilities"]]
            top_probabilities.append(max(eps, 1.0 - sum(top_probabilities) - eps))
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
            categories = [v for v in k_info["classes"]]
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
