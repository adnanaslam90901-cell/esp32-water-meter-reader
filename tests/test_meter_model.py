import cv2
import numpy as np

from app.meter_model import ctc_best_digits, mask_rect, normalize_rect, rectify, view_matrix


def rotated_strip_image(rect):
    """Photo with a strip that is dark on its left end and bright on its right end."""
    image = np.full((600, 800, 3), 128, np.uint8)
    strip = np.repeat(np.linspace(0, 255, 400, dtype=np.uint8)[None, :, None], 80, axis=0).repeat(3, axis=2)
    (cx, cy), _, angle = rect
    matrix = cv2.getRotationMatrix2D((200, 40), -angle, 1.0)
    matrix[:, 2] += [cx - 200, cy - 40]
    return cv2.warpAffine(strip, matrix, (800, 600), dst=image, borderMode=cv2.BORDER_TRANSPARENT)


def test_normalize_rect_long_side_first():
    (_, _), (w, h), angle = normalize_rect(((10, 10), (20, 100), 10))
    assert (w, h, angle) == (100, 20, 100 - 180)


def test_rectify_straightens_a_rotated_strip_and_flips():
    rect = ((400.0, 300.0), (400.0, 80.0), 30.0)
    image = rotated_strip_image(rect)
    crop = rectify(image, rect, pad_x=0, pad_y=0)
    assert crop.shape == (40, 192, 3)
    assert crop[:, :20].mean() + 150 < crop[:, -20:].mean()
    flipped = rectify(image, rect, pad_x=0, pad_y=0, flip=True)
    assert flipped[:, :20].mean() > flipped[:, -20:].mean() + 150


def test_mask_rect_maps_back_to_photo_coordinates():
    mask = np.zeros((1000, 1333), np.uint8)
    true_rect = ((700.0, 400.0), (300.0, 60.0), 20.0)
    cv2.fillPoly(mask, [cv2.boxPoints(true_rect).astype(np.int32)], 255)
    matrix = view_matrix((1333 / 2, 1000 / 2), (1333, 1000))
    prob = cv2.warpAffine(mask, matrix, (256, 256), flags=cv2.INTER_NEAREST).astype(np.float32) / 255
    (cx, cy), (w, h), angle = mask_rect(prob, matrix)[0]
    assert abs(cx - 700) < 8 and abs(cy - 400) < 8
    assert abs(w - 300) < 20 and abs(h - 60) < 12 and abs(angle - 20) < 4


def test_ctc_decoder_forces_digit_count_and_keeps_repeats():
    frames = [1, 0, 1, 0, 2, 2, 0, 3, 4, 5, 6, 7, 0, 0]  # blank-separated repeat: 0 _ 0
    log_probs = np.log(np.eye(11)[frames] * 0.98 + 0.002)
    assert ctc_best_digits(log_probs, 8)[0] == "00123456"
    assert len(ctc_best_digits(log_probs, 6)[0]) == 6
