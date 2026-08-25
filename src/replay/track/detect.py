"""Per-frame object detection.

Two implementations behind one protocol. ``ColorDetector`` needs no weights and
no network: it segments the green pitch, treats everything else inside the pitch
outline as a candidate object, and is what the synthetic demo clip runs on.
``YoloDetector`` is the real-footage path; ultralytics is imported inside its
constructor so that importing this module never pulls in torch.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Protocol, runtime_checkable

import cv2
import numpy as np

from replay.schema import Detection, ObjectClass

log = logging.getLogger(__name__)

# HSV bounds of the pitch. OpenCV hue is 0..179, so grass sits near 60. The
# saturation and value floors keep white lines and dark shadow out of the mask.
GREEN_LOWER = np.array([35, 60, 40], dtype=np.uint8)
GREEN_UPPER = np.array([85, 255, 255], dtype=np.uint8)

# A blob is the ball when it is small *and* washed out: the ball is white, so it
# has near-zero saturation and high value, unlike any jersey we care about.
BALL_MAX_SAT = 90.0
BALL_MIN_VAL = 140.0

# Colour cannot separate the ball from white shorts or a pitch marking — all
# three are the same white. Shape and context can. The ball is round, so it
# fills its bounding box like a disc does, and it is the only white thing that
# sits in open grass: shorts have a jersey above them and a marking fragment has
# more marking around it. Together these cut false balls by roughly four fifths
# on the demo clip at a cost of some recall, and the ones that get through are
# transient enough that the tracker's min_hits discards them.
BALL_MIN_FILL = 0.5
BALL_RING_GREEN = 0.7
BALL_RING_PAD = 4

# Shape sanity checks. Pitch lines survive the green mask as long thin blobs
# with a poor fill ratio, and stand/banner leakage shows up as one huge blob.
MAX_ASPECT = 6.0
MIN_FILL_RATIO = 0.15
MAX_AREA_FRACTION = 0.2

# Wide enough to bridge a pitch marking at broadcast resolution, narrow enough
# not to swallow the gap between two players standing close together.
PITCH_CLOSE_KERNEL = 15

# Markings are white like the ball, so they are told apart by thickness: this
# kernel must be wider than a line and narrower than the ball.
MARKING_OPEN_KERNEL = 5
WHITE_LOWER = np.array([0, 0, 150], dtype=np.uint8)
WHITE_UPPER = np.array([179, 80, 255], dtype=np.uint8)

# COCO ids ultralytics reports, mapped onto our two classes.
COCO_CLASS_MAP: dict[int, ObjectClass] = {0: "player", 32: "ball"}


@runtime_checkable
class Detector(Protocol):
    def detect(self, frame_bgr: np.ndarray, frame_idx: int) -> list[Detection]: ...


def resolve_device(device: str = "auto") -> str:
    """Resolve ``"auto"`` to the best available torch device.

    torch is an optional dependency (it only arrives with the ``yolo`` extra),
    so the import is local and a missing install degrades to CPU rather than
    raising: the colour detector has no use for a GPU anyway.
    """
    requested = device.strip().lower()
    if requested != "auto":
        return requested

    try:
        import torch
    except ImportError:
        log.debug("torch not installed; using cpu")
        return "cpu"

    try:
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception as exc:  # a backend probe failure must not be fatal
        log.debug("device probe failed (%s); using cpu", exc)
    return "cpu"


def _pitch_mask(green: np.ndarray) -> np.ndarray:
    """Fill the holes in the green mask to get the region the pitch occupies.

    Players punch holes in the grass mask, so the players themselves are exactly
    what is inside the green region but not green.

    The closing step before contour extraction is the part that matters. Pitch
    markings slice the grass into a dozen disconnected pieces, and taking
    external contours of that fragmented mask fills almost nothing: every player
    standing on or beside a line ends up outside all of them and is never seen.
    Closing bridges the markings first so the grass is a single region, and the
    largest filled contour is then the whole playing surface.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (PITCH_CLOSE_KERNEL,) * 2)
    closed = cv2.morphologyEx(green, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(green)
    if not contours:
        return filled
    cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    return filled


def _suppress_markings(hsv: np.ndarray, objects: np.ndarray) -> np.ndarray:
    """Remove pitch markings from the candidate-object mask.

    Markings are not green, so they survive into the object mask and weld
    themselves onto any player standing on them. They are separated from the
    ball by thickness rather than colour — both are white — so an opening with a
    kernel wider than a line keeps compact blobs and drops everything thin. What
    the opening removes is the markings; that is what gets subtracted.
    """
    white = cv2.inRange(hsv, WHITE_LOWER, WHITE_UPPER)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MARKING_OPEN_KERNEL,) * 2)
    compact = cv2.morphologyEx(white, cv2.MORPH_OPEN, kernel)
    thin = cv2.subtract(white, compact)
    # Dilate by one step so the anti-aliased shoulder of a line goes too,
    # otherwise a one-pixel fringe still bridges two players.
    thin = cv2.dilate(thin, np.ones((3, 3), np.uint8))
    return cv2.bitwise_and(objects, cv2.bitwise_not(thin))


def _in_open_grass(green: np.ndarray, x: int, y: int, bw: int, bh: int) -> bool:
    """True when the ring just outside a blob is mostly grass.

    A ball lies in open space; white shorts have a jersey directly above them
    and a leftover marking fragment has more marking beside it. Measuring the
    annulus rather than the blob is what makes the test work, since the blobs
    themselves are the same colour.
    """
    h, w = green.shape[:2]
    pad = BALL_RING_PAD
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
    ring = np.ones((y1 - y0, x1 - x0), dtype=bool)
    ring[(y - y0) : (y - y0 + bh), (x - x0) : (x - x0 + bw)] = False
    if ring.sum() < 8:
        return False
    return bool((green[y0:y1, x0:x1][ring] > 0).mean() >= BALL_RING_GREEN)


class ColorDetector:
    """Blob detector for solid-coloured objects on a green pitch."""

    def __init__(
        self, *, min_area: int = 30, ball_max_area: int = 120, ball_min_area: int = 8
    ) -> None:
        self.min_area = int(min_area)
        self.ball_max_area = int(ball_max_area)
        # The ball needs its own floor. A player-sized minimum is the right
        # noise gate for a torso but throws away the ball at the far end of the
        # pitch, which is only a dozen pixels across in a wide broadcast shot.
        self.ball_min_area = int(ball_min_area)

    def detect(self, frame_bgr: np.ndarray, frame_idx: int) -> list[Detection]:
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        frame = np.ascontiguousarray(frame_bgr)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"expected an HxWx3 BGR frame, got shape {frame.shape}")

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
        pitch = _pitch_mask(green)
        objects = _suppress_markings(hsv, cv2.bitwise_and(pitch, cv2.bitwise_not(green)))

        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(objects, 8)
        h, w = objects.shape[:2]
        max_area = MAX_AREA_FRACTION * float(h * w)

        sat = hsv[:, :, 1].astype(np.float64)
        val = hsv[:, :, 2].astype(np.float64)

        out: list[Detection] = []
        for label in range(1, n_labels):
            x, y, bw, bh, area = (int(v) for v in stats[label][:5])
            if area < min(self.min_area, self.ball_min_area) or area > max_area:
                continue
            if bw == 0 or bh == 0:
                continue
            aspect = max(bw / bh, bh / bw)
            if aspect > MAX_ASPECT:
                continue
            fill = area / float(bw * bh)
            if fill < MIN_FILL_RATIO:
                continue

            mask = labels[y : y + bh, x : x + bw] == label
            mean_sat = float(sat[y : y + bh, x : x + bw][mask].mean())
            mean_val = float(val[y : y + bh, x : x + bw][mask].mean())
            is_ball = (
                self.ball_min_area <= area <= self.ball_max_area
                and mean_sat <= BALL_MAX_SAT
                and mean_val >= BALL_MIN_VAL
                and fill >= BALL_MIN_FILL
                and _in_open_grass(green, x, y, bw, bh)
            )
            # Below the player floor the only thing worth keeping is a ball.
            if not is_ball and area < self.min_area:
                continue
            cls: ObjectClass = "ball" if is_ball else "player"

            # Score reports blob quality, not certainty: a crisp solid blob fills
            # its bounding box, a smeared or occluded one does not.
            score = float(np.clip(0.1 + 0.9 * fill, 0.0, 1.0))
            out.append(
                Detection(
                    frame=int(frame_idx),
                    x1=float(x),
                    y1=float(y),
                    x2=float(x + bw),
                    y2=float(y + bh),
                    score=score,
                    cls=cls,
                )
            )
        return out


def _as_numpy(value: Any) -> np.ndarray:
    """Detach a torch tensor if that is what ultralytics handed back."""
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy_fn = getattr(value, "numpy", None)
    if callable(numpy_fn):
        return np.asarray(numpy_fn())
    return np.asarray(value)


class YoloDetector:
    """ultralytics YOLO restricted to the person and sports-ball classes."""

    def __init__(
        self, *, weights: str = "yolov8n.pt", device: str = "auto", conf: float = 0.25
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise ImportError(
                "YoloDetector needs the optional 'yolo' extra: pip install 'replay[yolo]'"
            ) from exc

        self.weights = weights
        self.device = resolve_device(device)
        self.conf = float(conf)
        # Weights that are not on disk are fetched by ultralytics here, which is
        # the one place in the package that touches the network.
        self._model = YOLO(weights)

    def detect(self, frame_bgr: np.ndarray, frame_idx: int) -> list[Detection]:
        results = self._model.predict(frame_bgr, conf=self.conf, device=self.device, verbose=False)
        out: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = _as_numpy(boxes.xyxy).reshape(-1, 4)
            scores = _as_numpy(boxes.conf).reshape(-1)
            classes = _as_numpy(boxes.cls).reshape(-1).astype(int)
            for (x1, y1, x2, y2), score, coco in zip(xyxy, scores, classes, strict=True):
                cls = COCO_CLASS_MAP.get(int(coco))
                if cls is None:
                    continue
                out.append(
                    Detection(
                        frame=int(frame_idx),
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                        score=float(score),
                        cls=cls,
                    )
                )
        return out


def build_detector(name: Literal["color", "yolo"], **kwargs: Any) -> Detector:
    if name == "color":
        return ColorDetector(**kwargs)
    if name == "yolo":
        return YoloDetector(**kwargs)
    raise ValueError(f"unknown detector {name!r}; expected one of 'color', 'yolo'")
