import numpy as np
import streamlit as st

from PIL import Image
from typing import Tuple
from skimage.filters import gaussian
from skimage.filters import unsharp_mask
from skimage.transform import resize

from ..utils import sigmoid
from ..utils import to_uint8
from ..utils import resize_to
from ..utils import min_max_normalize
from ..utils import quantile_normalize
from ..utils import imagenet_normalize
from ..client import Client


def naive_cutout(normalized_img: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    if normalized_img.shape[-1] == 4:
        normalized_img = normalized_img[..., :3] * normalized_img[..., -1:]
    return to_uint8(np.concatenate([normalized_img, alpha[..., None]], axis=2))


def alpha_align(img: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    alpha_im = Image.fromarray(min_max_normalize(alpha))
    size = img.shape[1], img.shape[0]
    alpha = np.array(alpha_im.resize(size, Image.LANCZOS))
    alpha = np.clip(alpha, 0.0, 1.0)
    return alpha


def cutout(
    normalized_img: np.ndarray,
    alpha: np.ndarray,
    smooth: int = 0,
    tight: float = 0.9,
) -> Tuple[np.ndarray, np.ndarray]:
    alpha = alpha_align(normalized_img, alpha)
    if smooth > 0:
        alpha = gaussian(alpha, smooth)
        alpha = unsharp_mask(alpha, smooth, smooth * tight)
    alpha = quantile_normalize(alpha)
    rgba = naive_cutout(normalized_img, alpha)
    return alpha, rgba


@st.cache
def get_alpha(img: np.ndarray, client: Client, use_large: bool) -> np.ndarray:
    with st.spinner("Generating RGBA image..."):
        model = "sod.large" if use_large else "sod"
        logits = client.infer(model, {"input": img}).as_numpy("predictions")[0][0]
        logits = np.clip(logits, -50.0, 50.0)
        return sigmoid(logits)


def app(client: Client) -> None:
    use_large = st.sidebar.radio("Use large model", [True, False], index=1)
    use_threshold = st.sidebar.radio("Use threshold", [True, False], index=1)
    thresh = None
    if use_threshold:
        thresh = st.sidebar.slider("Threshold", min_value=0.0, max_value=1.0, value=0.5)
    smooth = st.sidebar.slider("Smooth", min_value=0, max_value=20, value=0)
    tight = st.sidebar.slider("Tight", min_value=0.0, max_value=1.0, value=0.9)

    uploaded_file = st.file_uploader("Please upload your file")
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        image = Image.open(uploaded_file).convert("RGB")
        original_shape = image.size
        original_normalized_img = np.array(image).astype(np.float32) / 255.0
        col1.image(image, caption="Uploaded Image")
        image.thumbnail((320, 320), Image.LANCZOS)
        img = resize(np.array(image), (320, 320), mode="constant")
        img = img[None, ...]
        img = min_max_normalize(img)
        img = imagenet_normalize(img)
        img = img.transpose([0, 3, 1, 2]).astype(np.float32)
        alpha = get_alpha(img, client, use_large)
        if thresh is not None:
            alpha = (alpha > thresh).astype(np.float32)
        alpha = min_max_normalize(alpha)
        alpha = np.array(resize_to(alpha, original_shape))
        alpha = alpha.astype(np.float32) / 255.0
        alpha, rgba = cutout(original_normalized_img, alpha, smooth, tight)
        col2.image(to_uint8(alpha), caption="Mask")
        st.image(rgba, caption="RGBA")
