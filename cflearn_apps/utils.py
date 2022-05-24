import numpy as np

from PIL import Image
from typing import Any


def min_max_normalize(arr: np.ndarray, *, global_norm: bool = True) -> np.ndarray:
    eps = 1.0e-8
    if global_norm:
        arr_min, arr_max = arr.min().item(), arr.max().item()
        return (arr - arr_min) / max(eps, arr_max - arr_min)
    arr_min, arr_max = arr.min(axis=0), arr.max(axis=0)
    diff = np.maximum(eps, arr_max - arr_min)
    return (arr - arr_min) / diff


def quantile_normalize(
    arr: np.ndarray,
    *,
    q: float = 0.01,
    global_norm: bool = True,
) -> np.ndarray:
    eps = 1.0e-8
    if global_norm:
        arr_min = np.quantile(arr, q).item()
        arr_max = np.quantile(arr, 1.0 - q).item()
        diff = max(eps, arr_max - arr_min)
    else:
        arr_min = np.quantile(arr, q, axis=0)
        arr_max = np.quantile(arr, 1.0 - q, axis=0)
        diff = np.maximum(eps, arr_max - arr_min)
    arr = np.clip(arr, arr_min, arr_max)
    return (arr - arr_min) / diff


def imagenet_normalize(arr: np.ndarray) -> np.ndarray:
    mean_gray, std_gray = [0.485], [0.229]
    mean_rgb, std_rgb = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    np_constructor = lambda inp: np.array(inp, dtype=np.float32).reshape([1, 1, 1, -1])
    if is_gray(arr):
        mean, std = map(np_constructor, [mean_gray, std_gray])
    else:
        mean, std = map(np_constructor, [mean_rgb, std_rgb])
    return (arr - mean) / std


def to_uint8(normalized_img: np.ndarray) -> np.ndarray:
    return (np.clip(normalized_img * 255.0, 0.0, 255.0)).astype(np.uint8)


def resize_to(normalized_img: np.ndarray, shape: Any) -> Image.Image:
    img = Image.fromarray(to_uint8(normalized_img))
    img = img.resize(shape, Image.LANCZOS)
    return img


def sigmoid(arr: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-arr))
