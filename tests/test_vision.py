import numpy as np

from app.vision import VisionEngine


def test_appearance_signature_is_normalized() -> None:
    frame = np.zeros((240, 160, 3), dtype=np.uint8)
    frame[20:110, 20:140] = (0, 0, 255)
    frame[110:210, 20:140] = (255, 0, 0)
    signature = VisionEngine._appearance_signature(frame, (10, 10, 150, 230))
    assert signature is not None
    # Two torso bands: 16x6 hue/saturation bins plus 8 brightness bins.
    assert len(signature) == 208
    assert abs(sum(signature) - 1.0) < 0.001
