from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image, ImageChops, ImageEnhance


@dataclass
class ELAResult:
    image: Image.Image
    score: int


def run_ela(image, quality=90):
    image = image.convert("RGB")

    compressed_bytes = BytesIO()
    image.save(compressed_bytes, "JPEG", quality=quality)
    compressed_bytes.seek(0)
    compressed = Image.open(compressed_bytes).convert("RGB")
    difference = ImageChops.difference(image, compressed)

    extrema = difference.getextrema()
    max_difference = max(channel[1] for channel in extrema) or 1
    scale = 255.0 / max_difference
    ela_image = ImageEnhance.Brightness(difference).enhance(scale)

    ela_array = np.array(ela_image)
    score = int(np.percentile(ela_array, 95))

    return ELAResult(image=ela_image, score=score)
