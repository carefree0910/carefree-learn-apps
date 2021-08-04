import os

import numpy as np

from PIL import Image
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Optional
from skimage import io
from skimage.filters import gaussian
from skimage.filters import unsharp_mask
from onnxruntime import InferenceSession
from torchvision.transforms import Compose

from .data import RescaleT
from .data import ToNormalizedArray
from ..toolkit import naive_cutout
from ..toolkit import min_max_normalize
from ..toolkit import alpha_matting_cutout
from ...constants import INPUT_KEY
from ...constants import WARNING_PREFIX


def cutout(
    img: np.ndarray,
    alpha: np.ndarray,
    smooth: int = 4,
    tight: float = 0.9,
    alpha_matting_config: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    alpha_im = Image.fromarray(min_max_normalize(alpha))
    alpha_im = alpha_im.resize((img.shape[1], img.shape[0]), Image.NEAREST)
    alpha = gaussian(np.array(alpha_im), smooth)
    alpha = unsharp_mask(alpha, smooth, smooth * tight)
    alpha = min_max_normalize(alpha)
    if alpha_matting_config is None:
        rgba = naive_cutout(img, alpha)
    else:
        try:
            rgba = alpha_matting_cutout(img, alpha, **alpha_matting_config)
        except Exception as err:
            print(
                f"{WARNING_PREFIX}alpha_matting failed ({err}), "
                f"naive cutting will be used"
            )
            rgba = naive_cutout(img, alpha)
    return alpha, rgba


def export(rgba: np.ndarray, tgt_path: Optional[str]) -> None:
    if tgt_path is not None:
        folder = os.path.split(tgt_path)[0]
        os.makedirs(folder, exist_ok=True)
        io.imsave(tgt_path, rgba)


class U2NetAPI:
    def __init__(
        self,
        onnx_path: str,
        rescale_size: int = 320,
    ):
        self.ort_session = InferenceSession(onnx_path)
        self.transform = Compose([RescaleT(rescale_size), ToNormalizedArray()])

    def _get_alpha(self, src: np.ndarray) -> np.ndarray:
        transformed = self.transform({INPUT_KEY: src})[INPUT_KEY][None, ...]
        ort_inputs = {
            node.name: transformed
            for node in self.ort_session.get_inputs()
        }
        logits = self.ort_session.run(None, ort_inputs)[0][0][0]
        logits = np.clip(logits, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def generate_cutout(
        self,
        src_path: str,
        tgt_path: Optional[str] = None,
        *,
        smooth: int = 16,
        tight: float = 0.5,
        alpha_matting_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        img = io.imread(src_path)
        img = img.astype(np.float32) / 255.0
        alpha = self._get_alpha(img)
        alpha, rgba = cutout(img, alpha, smooth, tight, alpha_matting_config)
        export(rgba, tgt_path)
        return alpha, rgba


__all__ = [
    "U2NetAPI",
]
