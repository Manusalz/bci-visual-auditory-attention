from __future__ import annotations

import ctypes
import math
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import mediapipe as mp

    MEDIAPIPE_AVAILABLE = True
except Exception:
    mp = None
    MEDIAPIPE_AVAILABLE = False

HAS_MP_SOLUTIONS = bool(MEDIAPIPE_AVAILABLE and hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"))
try:
    if MEDIAPIPE_AVAILABLE:
        from mediapipe.tasks import python as mp_tasks_python
        from mediapipe.tasks.python import vision as mp_tasks_vision

        HAS_MP_TASKS = True
    else:
        mp_tasks_python = None  # type: ignore[assignment]
        mp_tasks_vision = None  # type: ignore[assignment]
        HAS_MP_TASKS = False
except Exception:
    mp_tasks_python = None  # type: ignore[assignment]
    mp_tasks_vision = None  # type: ignore[assignment]
    HAS_MP_TASKS = False


@dataclass
class GazeEstimate:
    gaze_x: float
    gaze_y: float
    confidence: float
    face_found: bool
    eye_contours: list[np.ndarray] = field(default_factory=list)
    iris_points: list[tuple[int, int]] = field(default_factory=list)
    iris_circles: list[tuple[int, int, int]] = field(default_factory=list)
    pupil_points: list[tuple[int, int]] = field(default_factory=list)
    yaw_proxy: float = 0.0
    nose_tip: tuple[int, int] | None = None
    eye_midpoint: tuple[int, int] | None = None
    iris_distance_px: float = 0.0
    nose_to_eye_mid_px: float = 0.0


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _oriented_ratio(value: float, start: float, end: float) -> float:
    """
    Ratio preserving endpoint orientation.
    start -> 0, end -> 1 even if end < start.
    """
    den = end - start
    if abs(den) < 1e-6:
        return 0.5
    return _clip01((value - start) / den)


def _landmark_to_px(landmarks: list[Any], idx: int, width: int, height: int) -> tuple[float, float]:
    lm = landmarks[idx]
    return lm.x * width, lm.y * height


def _estimate_pupil_center(
    gray_frame: np.ndarray,
    eye_contour: np.ndarray,
    fallback_center: tuple[int, int],
) -> tuple[int, int]:
    if eye_contour.size == 0:
        return fallback_center

    contour_i32 = eye_contour.astype(np.int32)
    x, y, w, h = cv2.boundingRect(contour_i32)
    if w <= 2 or h <= 2:
        return fallback_center

    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(gray_frame.shape[1], x + w)
    y1 = min(gray_frame.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return fallback_center

    roi = gray_frame[y0:y1, x0:x1]
    if roi.size == 0:
        return fallback_center

    local_contour = contour_i32.copy()
    local_contour[:, 0] -= x0
    local_contour[:, 1] -= y0
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [local_contour], 255)

    valid = mask > 0
    if not np.any(valid):
        return fallback_center

    blur = cv2.GaussianBlur(roi, (5, 5), 0)
    eye_values = blur[valid]
    if eye_values.size < 10:
        return fallback_center

    dark_thr = np.percentile(eye_values, 20)
    candidates = (blur <= dark_thr) & valid
    ys, xs = np.where(candidates)
    if xs.size == 0:
        return fallback_center

    candidate_values = blur[ys, xs].astype(np.float64)
    weights = (dark_thr + 1.0) - candidate_values
    weights = np.clip(weights, 1e-3, None)

    cx_local = float(np.average(xs, weights=weights))
    cy_local = float(np.average(ys, weights=weights))
    cx = int(np.clip(round(x0 + cx_local), 0, gray_frame.shape[1] - 1))
    cy = int(np.clip(round(y0 + cy_local), 0, gray_frame.shape[0] - 1))
    return cx, cy


def _iris_circle_from_points(iris_points: list[tuple[float, float]]) -> tuple[int, int, int]:
    arr = np.asarray(iris_points, dtype=np.float32)
    (cx, cy), radius = cv2.minEnclosingCircle(arr)
    r = max(1, int(round(radius)))
    return int(round(cx)), int(round(cy)), r


def _estimate_pupil_from_iris(
    gray_frame: np.ndarray,
    iris_circle: tuple[int, int, int],
    fallback_center: tuple[int, int],
) -> tuple[int, int]:
    cx, cy, radius = iris_circle
    if radius <= 1:
        return fallback_center

    x0 = max(0, cx - radius)
    y0 = max(0, cy - radius)
    x1 = min(gray_frame.shape[1], cx + radius + 1)
    y1 = min(gray_frame.shape[0], cy + radius + 1)
    if x1 <= x0 or y1 <= y0:
        return fallback_center

    roi = gray_frame[y0:y1, x0:x1]
    if roi.size == 0:
        return fallback_center

    local_cx = cx - x0
    local_cy = cy - y0
    inner_r = max(1, int(round(radius * 0.85)))
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (local_cx, local_cy), inner_r, 255, -1)

    valid = mask > 0
    if not np.any(valid):
        return fallback_center

    blur = cv2.GaussianBlur(roi, (5, 5), 0)
    values = blur[valid]
    if values.size < 10:
        return fallback_center

    dark_thr = np.percentile(values, 25)
    candidates = (blur <= dark_thr) & valid
    ys, xs = np.where(candidates)
    if xs.size == 0:
        return fallback_center

    candidate_values = blur[ys, xs].astype(np.float64)
    weights = (dark_thr + 1.0) - candidate_values
    weights = np.clip(weights, 1e-3, None)

    dark_x = float(np.average(xs, weights=weights))
    dark_y = float(np.average(ys, weights=weights))

    # Keep the pupil estimate close to iris center for stability.
    blend = 0.6
    px_local = (1.0 - blend) * local_cx + blend * dark_x
    py_local = (1.0 - blend) * local_cy + blend * dark_y

    dx = px_local - local_cx
    dy = py_local - local_cy
    max_disp = radius * 0.45
    dist = math.hypot(dx, dy)
    if dist > max_disp and dist > 1e-6:
        scale = max_disp / dist
        px_local = local_cx + dx * scale
        py_local = local_cy + dy * scale

    px = int(np.clip(round(x0 + px_local), 0, gray_frame.shape[1] - 1))
    py = int(np.clip(round(y0 + py_local), 0, gray_frame.shape[0] - 1))
    return px, py


def _ensure_task_model(model_path: Path, model_url: str, auto_download: bool) -> Path:
    if model_path.exists():
        return model_path
    if not auto_download:
        raise RuntimeError(f"Face Landmarker model not found: {model_path}")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(model_url, model_path.as_posix())
    if not model_path.exists() or model_path.stat().st_size <= 0:
        raise RuntimeError(f"Failed to download model: {model_path}")
    return model_path


class MediaPipeGazeTracker:
    LEFT_EYE_OUTER = 33
    LEFT_EYE_INNER = 133
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145

    RIGHT_EYE_OUTER = 263
    RIGHT_EYE_INNER = 362
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374

    # Pair iris indices with corresponding eye-corner indices used below.
    # In current MediaPipe wheels, 469-472 align with the 33/133 eye side,
    # and 474-477 align with the 263/362 eye side.
    LEFT_IRIS = (469, 470, 471, 472)
    RIGHT_IRIS = (474, 475, 476, 477)

    LEFT_EYE_CONTOUR = (33, 160, 158, 133, 153, 144)
    RIGHT_EYE_CONTOUR = (263, 387, 385, 362, 380, 373)

    # Nose tip landmark — used for head-pose yaw estimation.
    NOSE_TIP = 1

    DEFAULT_TASK_MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    )

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        backend: str = "auto",
        task_model_path: str | None = None,
        task_model_url: str | None = None,
        auto_download_task_model: bool = True,
        head_pose_correction: bool = False,
        pose_gain_x: float = 0.3,
        pose_dead_zone: float = 0.0,
    ) -> None:
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("mediapipe not available")

        requested = str(backend).lower()
        if requested not in {"auto", "solutions", "tasks"}:
            raise ValueError(f"Invalid MediaPipe backend: {backend}")

        if requested == "solutions":
            if not HAS_MP_SOLUTIONS:
                raise RuntimeError("MediaPipe solutions backend unavailable in this installation.")
            chosen = "solutions"
        elif requested == "tasks":
            if not HAS_MP_TASKS:
                raise RuntimeError("MediaPipe tasks backend unavailable in this installation.")
            chosen = "tasks"
        else:
            chosen = "solutions" if HAS_MP_SOLUTIONS else "tasks"
            if chosen == "tasks" and not HAS_MP_TASKS:
                raise RuntimeError("No supported MediaPipe backend found (solutions/tasks unavailable).")

        self.backend = chosen
        self._face_mesh = None
        self._landmarker = None
        self.head_pose_correction = bool(head_pose_correction)
        self.pose_gain_x = float(pose_gain_x)
        self.pose_dead_zone = float(pose_dead_zone)

        if self.backend == "solutions":
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        else:
            model_rel = task_model_path or str(Path(__file__).resolve().parent / "models" / "face_landmarker.task")
            model_path = Path(model_rel)
            if not model_path.is_absolute():
                model_path = (Path.cwd() / model_path).resolve()
            model_url = task_model_url or self.DEFAULT_TASK_MODEL_URL
            model_path = _ensure_task_model(model_path, model_url, auto_download_task_model)

            base_options = mp_tasks_python.BaseOptions(model_asset_path=str(model_path))
            options = mp_tasks_vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_tasks_vision.RunningMode.VIDEO,
                num_faces=1,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self._landmarker = mp_tasks_vision.FaceLandmarker.create_from_options(options)

    def _compute_from_landmarks(self, landmarks: list[Any], frame_bgr: np.ndarray) -> GazeEstimate:
        frame_h, frame_w = frame_bgr.shape[:2]
        max_required = max(
            self.LEFT_EYE_OUTER,
            self.LEFT_EYE_INNER,
            self.LEFT_EYE_TOP,
            self.LEFT_EYE_BOTTOM,
            self.RIGHT_EYE_OUTER,
            self.RIGHT_EYE_INNER,
            self.RIGHT_EYE_TOP,
            self.RIGHT_EYE_BOTTOM,
            max(self.LEFT_IRIS),
            max(self.RIGHT_IRIS),
        )
        if len(landmarks) <= max_required:
            return GazeEstimate(0.5, 0.5, 0.0, False, [], [], [], [])

        left_outer = _landmark_to_px(landmarks, self.LEFT_EYE_OUTER, frame_w, frame_h)
        left_inner = _landmark_to_px(landmarks, self.LEFT_EYE_INNER, frame_w, frame_h)
        left_top = _landmark_to_px(landmarks, self.LEFT_EYE_TOP, frame_w, frame_h)
        left_bottom = _landmark_to_px(landmarks, self.LEFT_EYE_BOTTOM, frame_w, frame_h)

        right_outer = _landmark_to_px(landmarks, self.RIGHT_EYE_OUTER, frame_w, frame_h)
        right_inner = _landmark_to_px(landmarks, self.RIGHT_EYE_INNER, frame_w, frame_h)
        right_top = _landmark_to_px(landmarks, self.RIGHT_EYE_TOP, frame_w, frame_h)
        right_bottom = _landmark_to_px(landmarks, self.RIGHT_EYE_BOTTOM, frame_w, frame_h)

        left_iris_pts = [_landmark_to_px(landmarks, i, frame_w, frame_h) for i in self.LEFT_IRIS]
        right_iris_pts = [_landmark_to_px(landmarks, i, frame_w, frame_h) for i in self.RIGHT_IRIS]

        left_iris_center = np.mean(np.asarray(left_iris_pts), axis=0)
        right_iris_center = np.mean(np.asarray(right_iris_pts), axis=0)

        # Keep orientation by eye (outer->inner, top->bottom). This avoids
        # horizontal cancellation between eyes.
        left_x = _oriented_ratio(float(left_iris_center[0]), left_outer[0], left_inner[0])
        right_x = _oriented_ratio(float(right_iris_center[0]), right_outer[0], right_inner[0])
        left_y = _oriented_ratio(float(left_iris_center[1]), left_top[1], left_bottom[1])
        right_y = _oriented_ratio(float(right_iris_center[1]), right_top[1], right_bottom[1])

        gaze_x = _clip01((left_x + right_x) / 2.0)
        gaze_y = _clip01((left_y + right_y) / 2.0)

        # Head pose yaw compensation: nose tip displacement relative to eye midpoint.
        # yaw_proxy > 0 when nose is to the right of eye midpoint (head turns right).
        # pose_gain_x can be positive or negative — tune empirically with known gaze targets.
        yaw_proxy = 0.0
        nose_tip = None
        eye_midpoint = (
            int(round((left_iris_center[0] + right_iris_center[0]) / 2.0)),
            int(round((left_iris_center[1] + right_iris_center[1]) / 2.0)),
        )
        iris_distance_px = float(math.hypot(right_iris_center[0] - left_iris_center[0], right_iris_center[1] - left_iris_center[1]))
        nose_to_eye_mid_px = 0.0
        if len(landmarks) > self.NOSE_TIP:
            nose_x, nose_y = _landmark_to_px(landmarks, self.NOSE_TIP, frame_w, frame_h)
            nose_tip = (int(round(nose_x)), int(round(nose_y)))
            eye_mid_x = (left_iris_center[0] + right_iris_center[0]) / 2.0
            face_width_px = max(abs(right_outer[0] - left_outer[0]), 1.0)
            yaw_proxy = (nose_x - eye_mid_x) / face_width_px
            nose_to_eye_mid_px = float(math.hypot(nose_x - float(eye_midpoint[0]), nose_y - float(eye_midpoint[1])))
            if self.head_pose_correction and abs(yaw_proxy) > self.pose_dead_zone:
                # Solo corregir si el giro de cabeza supera la zona muerta
                effective_yaw = yaw_proxy - math.copysign(self.pose_dead_zone, yaw_proxy)
                gaze_x = _clip01(gaze_x + self.pose_gain_x * effective_yaw)

        left_span = abs(left_inner[0] - left_outer[0]) / max(frame_w, 1)
        right_span = abs(right_inner[0] - right_outer[0]) / max(frame_w, 1)
        span = (left_span + right_span) / 2.0
        confidence = float(np.clip((span - 0.01) / 0.03, 0.0, 1.0))

        left_contour = np.asarray(
            [_landmark_to_px(landmarks, i, frame_w, frame_h) for i in self.LEFT_EYE_CONTOUR],
            dtype=np.int32,
        )
        right_contour = np.asarray(
            [_landmark_to_px(landmarks, i, frame_w, frame_h) for i in self.RIGHT_EYE_CONTOUR],
            dtype=np.int32,
        )
        iris_points = [
            (int(left_iris_center[0]), int(left_iris_center[1])),
            (int(right_iris_center[0]), int(right_iris_center[1])),
        ]
        left_iris_circle = _iris_circle_from_points(left_iris_pts)
        right_iris_circle = _iris_circle_from_points(right_iris_pts)
        iris_circles = [left_iris_circle, right_iris_circle]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        pupil_points = [
            _estimate_pupil_from_iris(gray, left_iris_circle, iris_points[0]),
            _estimate_pupil_from_iris(gray, right_iris_circle, iris_points[1]),
        ]
        return GazeEstimate(
            gaze_x,
            gaze_y,
            confidence,
            True,
            [left_contour, right_contour],
            iris_points,
            iris_circles,
            pupil_points,
            yaw_proxy=yaw_proxy,
            nose_tip=nose_tip,
            eye_midpoint=eye_midpoint,
            iris_distance_px=iris_distance_px,
            nose_to_eye_mid_px=nose_to_eye_mid_px,
        )

    def _estimate_solutions(self, frame_bgr: np.ndarray) -> GazeEstimate:
        result = self._face_mesh.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not result.multi_face_landmarks:
            return GazeEstimate(0.5, 0.5, 0.0, False, [], [], [], [])
        landmarks = result.multi_face_landmarks[0].landmark
        return self._compute_from_landmarks(landmarks, frame_bgr)

    def _estimate_tasks(self, frame_bgr: np.ndarray, now_s: float) -> GazeEstimate:
        frame_h, frame_w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(now_s * 1000)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.face_landmarks:
            return GazeEstimate(0.5, 0.5, 0.0, False, [], [], [], [])
        landmarks = result.face_landmarks[0]
        return self._compute_from_landmarks(landmarks, frame_bgr)

    def estimate(self, frame_bgr: np.ndarray, now_s: float | None = None) -> GazeEstimate:
        if now_s is None:
            now_s = time.time()
        if frame_bgr is None:
            return GazeEstimate(0.5, 0.5, 0.0, False, [], [], [], [])
        if self.backend == "solutions":
            return self._estimate_solutions(frame_bgr)
        return self._estimate_tasks(frame_bgr, now_s=now_s)

    def close(self) -> None:
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None


class SimulatedGazeTracker:
    """Lightweight synthetic source for demo/testing without webcam."""

    def __init__(self) -> None:
        self._t0 = time.time()

    def estimate(self, frame_bgr: np.ndarray | None, now_s: float | None = None) -> GazeEstimate:
        if now_s is None:
            now_s = time.time()
        t = now_s - self._t0
        phase = t % 9.0
        burst = 0.2 if 5.5 <= phase <= 6.4 else 0.0
        gaze_x = _clip01(0.5 + 0.04 * math.sin(2.0 * math.pi * 0.2 * t) + burst)
        gaze_y = _clip01(0.5 + 0.03 * math.sin(2.0 * math.pi * 0.27 * t + 1.2))
        return GazeEstimate(gaze_x, gaze_y, 0.95, True, [], [], [], [])

    def close(self) -> None:
        return None


class _Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class MouseGazeTracker:
    """Uses cursor position as gaze proxy."""

    def __init__(self) -> None:
        self._user32 = ctypes.windll.user32

    def estimate(self, frame_bgr: np.ndarray | None, now_s: float | None = None) -> GazeEstimate:
        if now_s is None:
            now_s = time.time()
        _ = now_s

        pt = _Point()
        self._user32.GetCursorPos(ctypes.byref(pt))
        screen_w = max(1, int(self._user32.GetSystemMetrics(0)))
        screen_h = max(1, int(self._user32.GetSystemMetrics(1)))
        gaze_x = _clip01(pt.x / screen_w)
        gaze_y = _clip01(pt.y / screen_h)
        return GazeEstimate(gaze_x, gaze_y, 1.0, True, [], [], [], [])

    def close(self) -> None:
        return None
