import numpy as np

from PIL import Image
from typing import Any
from typing import List
from scipy.ndimage.morphology import binary_erosion
from pymatting.util.util import stack_images
from pymatting.alpha.estimate_alpha_cf import estimate_alpha_cf
from pymatting.foreground.estimate_foreground_ml import estimate_foreground_ml


def is_gray(arr: np.ndarray) -> bool:
    if isinstance(arr, np.ndarray):
        return arr.shape[-1] == 1
    if len(arr.shape) == 3:
        return arr.shape[0] == 1
    return arr.shape[1] == 1


def min_max_normalize(arr: np.ndarray, *, global_norm: bool = True) -> np.ndarray:
    eps = 1.0e-8
    if global_norm:
        arr_min, arr_max = arr.min().item(), arr.max().item()
        return (arr - arr_min) / max(eps, arr_max - arr_min)
    arr_min, arr_max = arr.min(axis=0), arr.max(axis=0)
    diff = np.maximum(eps, arr_max - arr_min)
    return (arr - arr_min) / diff


def imagenet_normalize(arr: np.ndarray) -> np.ndarray:
    mean_gray, std_gray = [0.485], [0.229]
    mean_rgb, std_rgb = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    np_constructor = lambda inp: np.array(inp, dtype=np.float32).reshape([1, 1, -1])
    if is_gray(arr):
        mean, std = map(np_constructor, [mean_gray, std_gray])
    else:
        mean, std = map(np_constructor, [mean_rgb, std_rgb])
    return (arr - mean) / std


def to_uint8(normalized: np.ndarray) -> np.ndarray:
    return (np.clip(normalized * 255.0, 0.0, 255.0)).astype(np.uint8)


def naive_cutout(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if img.shape[-1] == 4:
        img = img[..., :3] * img[..., -1:]
    return to_uint8(np.concatenate([img, mask[..., None]], axis=2))


def alpha_matting_cutout(
    img: np.ndarray,
    mask: np.ndarray,
    foreground_threshold: float = 0.94117647,
    background_threshold: float = 0.0392156862745,
    erode_structure_size: int = 10,
    base_size: int = 1000,
) -> np.ndarray:
    img_ins = Image.fromarray(to_uint8(img)).convert("RGB")
    size = img_ins.size
    img_ins.thumbnail((base_size, base_size), Image.LANCZOS)
    mask_ins = Image.fromarray(mask)
    mask_ins = mask_ins.resize(img_ins.size, Image.LANCZOS)

    img = np.array(img_ins)
    mask = np.array(mask_ins)

    is_foreground = mask > foreground_threshold
    is_background = mask < background_threshold

    structure = None
    if erode_structure_size > 0:
        structure = np.ones((erode_structure_size, erode_structure_size), dtype=np.int)

    is_foreground = binary_erosion(is_foreground, structure=structure)
    is_background = binary_erosion(is_background, structure=structure, border_value=1)

    trimap = np.full(mask.shape, dtype=np.uint8, fill_value=128)
    trimap[is_foreground] = 255
    trimap[is_background] = 0

    img_normalized = img / 255.0
    trimap_normalized = trimap / 255.0

    alpha = estimate_alpha_cf(img_normalized, trimap_normalized)
    foreground = estimate_foreground_ml(img_normalized, alpha)
    cutout = stack_images(foreground, alpha)

    cutout = to_uint8(cutout)
    cutout = Image.fromarray(cutout)
    cutout = cutout.resize(size, Image.LANCZOS)

    return np.array(cutout)


class Compose:
    def __init__(self, transforms: List[Any]):
        self.transforms = transforms

    def __call__(self, inp: Any) -> Any:
        for t in self.transforms:
            inp = t(inp)
        return inp

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
