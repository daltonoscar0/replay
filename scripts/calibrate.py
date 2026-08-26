"""Make a camera calibration file for a real broadcast clip.

`replay track` needs a homography to turn pixels into pitch meters. It will try
the SoccerNet model and then line fitting on its own, but line fitting is
conservative and gives up on a lot of real frames rather than guess. This is the
fallback for those: click four or more known pitch landmarks and it solves for
the camera.

    python scripts/calibrate.py match.mp4 --out camera.json

Click each landmark it names, in order, then press `s` to solve. `u` undoes the
last click, `n` skips a landmark you cannot see, `q` quits. It always writes a
check image showing the pitch model drawn back onto the frame, because a
homography that is subtly wrong looks fine as nine numbers and obvious as a
picture.

Without a display, or to script it, supply the correspondences directly:

    python scripts/calibrate.py match.mp4 --pairs pairs.json --out camera.json

where pairs.json is {"landmark_name": [u, v], ...} in image pixels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from replay.track.homography import PITCH_KEYPOINTS, Homography

# Ordered so the easiest, most widely separated landmarks come first. A
# homography is only as good as the spread of its correspondences: four points
# clustered in one corner of the frame will fit perfectly and project terribly
# everywhere else.
SUGGESTED = [
    "halfway_top",
    "halfway_bottom",
    "center_circle_top",
    "center_circle_bottom",
    "penalty_box_left_front_top",
    "penalty_box_left_front_bottom",
    "penalty_box_right_front_top",
    "penalty_box_right_front_bottom",
    "corner_left_top",
    "corner_left_bottom",
    "corner_right_top",
    "corner_right_bottom",
    "penalty_spot_left",
    "penalty_spot_right",
    "six_yard_left_front_top",
    "six_yard_left_front_bottom",
    "six_yard_right_front_top",
    "six_yard_right_front_bottom",
]

PITCH_LINES = [
    ("corner_left_bottom", "corner_right_bottom"),
    ("corner_left_top", "corner_right_top"),
    ("corner_left_bottom", "corner_left_top"),
    ("corner_right_bottom", "corner_right_top"),
    ("halfway_bottom", "halfway_top"),
    ("penalty_box_left_goal_bottom", "penalty_box_left_front_bottom"),
    ("penalty_box_left_front_bottom", "penalty_box_left_front_top"),
    ("penalty_box_left_front_top", "penalty_box_left_goal_top"),
    ("penalty_box_right_goal_bottom", "penalty_box_right_front_bottom"),
    ("penalty_box_right_front_bottom", "penalty_box_right_front_top"),
    ("penalty_box_right_front_top", "penalty_box_right_goal_top"),
    ("six_yard_left_goal_bottom", "six_yard_left_front_bottom"),
    ("six_yard_left_front_bottom", "six_yard_left_front_top"),
    ("six_yard_left_front_top", "six_yard_left_goal_top"),
    ("six_yard_right_goal_bottom", "six_yard_right_front_bottom"),
    ("six_yard_right_front_bottom", "six_yard_right_front_top"),
    ("six_yard_right_front_top", "six_yard_right_goal_top"),
]


def read_frame(video: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"could not open {video}")
    if index:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read frame {index} of {video}")
    return frame


def solve(picked: dict[str, tuple[float, float]]) -> Homography:
    if len(picked) < 4:
        raise SystemExit(
            f"need at least 4 landmarks to fit a homography, got {len(picked)}"
        )
    names = sorted(picked)
    image_pts = np.array([picked[n] for n in names], dtype=np.float64)
    pitch_pts = np.array([PITCH_KEYPOINTS[n] for n in names], dtype=np.float64)
    return Homography.from_point_pairs(image_pts, pitch_pts)


def draw_check(frame: np.ndarray, h: Homography, out: Path) -> Path:
    """Project the pitch model back into the frame so the fit can be eyeballed."""
    canvas = frame.copy()
    for a, b in PITCH_LINES:
        pa, pb = PITCH_KEYPOINTS[a], PITCH_KEYPOINTS[b]
        ia, ib = h.pitch_to_image(np.array([pa, pb], dtype=np.float64))
        if not (np.isfinite(ia).all() and np.isfinite(ib).all()):
            continue
        cv2.line(canvas, tuple(np.round(ia).astype(int)), tuple(np.round(ib).astype(int)),
                 (0, 255, 255), 2, cv2.LINE_AA)
    for name in ("center_spot", "penalty_spot_left", "penalty_spot_right"):
        p = h.pitch_to_image(np.array([PITCH_KEYPOINTS[name]], dtype=np.float64))[0]
        if np.isfinite(p).all():
            cv2.circle(canvas, tuple(np.round(p).astype(int)), 5, (0, 255, 255), -1, cv2.LINE_AA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    return out


def pick_interactively(frame: np.ndarray) -> dict[str, tuple[float, float]]:
    picked: dict[str, tuple[float, float]] = {}
    order = list(SUGGESTED)
    idx = 0
    click: list[tuple[float, float]] = []

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            click.append((float(x), float(y)))

    window = "replay calibrate - click the named landmark"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    print("click each landmark as it is named.  s=solve  u=undo  n=skip  q=quit")

    while True:
        view = frame.copy()
        for name, (u, v) in picked.items():
            cv2.circle(view, (int(u), int(v)), 5, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.putText(view, name, (int(u) + 8, int(v) - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 255, 0), 1, cv2.LINE_AA)
        target = order[idx] if idx < len(order) else None
        banner = (
            f"click: {target}   ({len(picked)} picked, 4 needed)"
            if target
            else f"{len(picked)} picked - press s to solve"
        )
        cv2.rectangle(view, (0, 0), (view.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(view, banner, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(window, view)

        if click and target is not None:
            picked[target] = click.pop()
            idx += 1
            continue
        click.clear()

        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            raise SystemExit("cancelled")
        if key == ord("s"):
            break
        if key == ord("n") and target is not None:
            idx += 1
        if key == ord("u") and picked:
            last = list(picked)[-1]
            del picked[last]
            idx = max(0, order.index(last)) if last in order else max(0, idx - 1)

    cv2.destroyAllWindows()
    return picked


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, default=Path("camera.json"))
    ap.add_argument("--frame", type=int, default=0, help="Frame to calibrate on.")
    ap.add_argument("--pairs", type=Path, help='{"landmark": [u, v], ...} instead of clicking.')
    ap.add_argument("--check", type=Path, help="Where to write the check image.")
    args = ap.parse_args(argv)

    frame = read_frame(args.video, args.frame)

    if args.pairs:
        raw = json.loads(args.pairs.read_text())
        unknown = sorted(set(raw) - set(PITCH_KEYPOINTS))
        if unknown:
            raise SystemExit(
                f"unknown landmark(s): {', '.join(unknown)}\n"
                f"valid names: {', '.join(sorted(PITCH_KEYPOINTS))}"
            )
        picked = {k: (float(v[0]), float(v[1])) for k, v in raw.items()}
    else:
        picked = pick_interactively(frame)

    h = solve(picked)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    h.to_json(args.out)

    check = args.check or args.out.with_name(args.out.stem + "_check.png")
    draw_check(frame, h, check)

    err = h.reprojection_error
    print(f"wrote {args.out} from {len(picked)} landmarks")
    print(f"reprojection error: {err:.2f} m")
    print(f"check image: {check}  <- look at this before trusting the calibration")
    if not np.isfinite(err) or err > 2.0:
        print(
            "\nWARNING: that error is high. Usually it means a landmark was clicked in\n"
            "the wrong place, or every point sits in one part of the frame. Spread the\n"
            "picks across the visible pitch and re-check the image.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
