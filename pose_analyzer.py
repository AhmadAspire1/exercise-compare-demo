import cv2
import numpy as np
import mediapipe as mp

from metrics import angle_3pts, ema

# MediaPipe indices (PoseLandmark)
# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
LM = mp.solutions.pose.PoseLandmark

KEYPOINTS = {
    "LS": LM.LEFT_SHOULDER.value,
    "RS": LM.RIGHT_SHOULDER.value,
    "LE": LM.LEFT_ELBOW.value,
    "RE": LM.RIGHT_ELBOW.value,
    "LW": LM.LEFT_WRIST.value,
    "RW": LM.RIGHT_WRIST.value,
    "LH": LM.LEFT_HIP.value,
    "RH": LM.RIGHT_HIP.value,
    "LK": LM.LEFT_KNEE.value,
    "RK": LM.RIGHT_KNEE.value,
    "LA": LM.LEFT_ANKLE.value,
    "RA": LM.RIGHT_ANKLE.value,
}

def _extract_landmarks_from_video(video_path: str, sample_fps: int = 10, min_visibility: float = 0.4):
    """
    Returns:
      frames_ts: list of timestamps in seconds
      lms: numpy array shape (T, 33, 3) in normalized coords [0..1] for x,y plus visibility
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 1e-3 else 30.0
    stride = max(1, int(round(fps / sample_fps)))

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frames_ts = []
    out = []

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        ts = frame_idx / fps
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if not res.pose_landmarks:
            frame_idx += 1
            continue

        lms = []
        for lm in res.pose_landmarks.landmark:
            lms.append([lm.x, lm.y, lm.visibility])
        lms = np.array(lms, dtype=np.float32)  # (33, 3)

        # Filter low visibility frames
        needed = [KEYPOINTS[k] for k in ["LS","RS","LH","RH","LK","RK","LA","RA"]]
        vis = lms[needed, 2]
        if float(np.min(vis)) < min_visibility:
            frame_idx += 1
            continue

        frames_ts.append(ts)
        out.append(lms)
        frame_idx += 1

    cap.release()
    pose.close()

    if len(out) < 10:
        raise RuntimeError("Not enough valid pose frames detected. Try a clearer full-body video.")

    return np.array(frames_ts, dtype=np.float32), np.stack(out, axis=0)

def _get_xy(lms: np.ndarray, name: str):
    idx = KEYPOINTS[name]
    return lms[:, idx, 0:2]  # (T,2)

def _compute_features(lms: np.ndarray):
    """
    Build per-frame feature vectors and interpretable metrics.

    Returns dict:
      feat_seq: (T, D)
      series: dict of named time series (T,)
      summary: dict of aggregated stats
    """
    # keypoint tracks
    LS = _get_xy(lms, "LS")
    RS = _get_xy(lms, "RS")
    LH = _get_xy(lms, "LH")
    RH = _get_xy(lms, "RH")
    LK = _get_xy(lms, "LK")
    RK = _get_xy(lms, "RK")
    LA = _get_xy(lms, "LA")
    RA = _get_xy(lms, "RA")

    # Midpoints
    SH_mid = (LS + RS) / 2.0
    HIP_mid = (LH + RH) / 2.0
    KNEE_mid = (LK + RK) / 2.0
    ANK_mid = (LA + RA) / 2.0

    # Knee angles
    knee_L = np.array([angle_3pts(LH[i], LK[i], LA[i]) for i in range(len(lms))], dtype=np.float32)
    knee_R = np.array([angle_3pts(RH[i], RK[i], RA[i]) for i in range(len(lms))], dtype=np.float32)
    knee_mean = (knee_L + knee_R) / 2.0

    # Hip angles (shoulder-hip-knee)
    hip_L = np.array([angle_3pts(LS[i], LH[i], LK[i]) for i in range(len(lms))], dtype=np.float32)
    hip_R = np.array([angle_3pts(RS[i], RH[i], RK[i]) for i in range(len(lms))], dtype=np.float32)
    hip_mean = (hip_L + hip_R) / 2.0

    # Trunk lean proxy: horizontal offset between shoulder-mid and hip-mid
    trunk_lean = np.abs(SH_mid[:, 0] - HIP_mid[:, 0]).astype(np.float32)

    # Knee valgus proxy: knee-mid x relative to ankle-mid x (how much knees move inward)
    valgus = np.abs(KNEE_mid[:, 0] - ANK_mid[:, 0]).astype(np.float32)

    # Vertical signals for rep counting (hip or shoulder movement)
    hip_y = HIP_mid[:, 1].astype(np.float32)
    sh_y = SH_mid[:, 1].astype(np.float32)

    # Smooth series
    knee_mean_s = ema(knee_mean, 0.2)
    hip_mean_s = ema(hip_mean, 0.2)
    trunk_s = ema(trunk_lean, 0.2)
    valgus_s = ema(valgus, 0.2)
    hip_y_s = ema(hip_y, 0.2)
    sh_y_s = ema(sh_y, 0.2)

    # Feature vector per frame
    # Normalize some components roughly to similar scales
    feat = np.stack([
        (knee_mean_s / 180.0),
        (hip_mean_s / 180.0),
        trunk_s,   # already 0..1-ish
        valgus_s,  # 0..1-ish
        hip_y_s,   # 0..1
        sh_y_s     # 0..1
    ], axis=1).astype(np.float32)

    summary = {
        "knee_angle_mean_deg": float(np.mean(knee_mean_s)),
        "knee_angle_min_deg": float(np.min(knee_mean_s)),
        "hip_angle_mean_deg": float(np.mean(hip_mean_s)),
        "trunk_lean_mean": float(np.mean(trunk_s)),
        "valgus_mean": float(np.mean(valgus_s)),
        "depth_proxy": float(180.0 - np.min(knee_mean_s)),  # higher => deeper (proxy)
    }

    series = {
        "knee_angle_deg": knee_mean_s,
        "hip_angle_deg": hip_mean_s,
        "trunk_lean": trunk_s,
        "valgus": valgus_s,
        "hip_y": hip_y_s,
        "shoulder_y": sh_y_s,
    }
    return feat, series, summary

def _count_reps_from_signal(signal: np.ndarray, min_prominence: float = 0.01):
    """
    Very simple peak counting:
    Reps are counted by local maxima of hip_y (down movement increases y in image coords).
    We count cycles by finding peaks and requiring minimum spacing.

    This is demo-level (works well with stable camera + full body).
    """
    x = signal.astype(np.float32)
    # normalize
    x = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-9)

    peaks = []
    for i in range(1, len(x)-1):
        if x[i] > x[i-1] and x[i] > x[i+1]:
            peaks.append(i)

    # filter by prominence-ish: peak must be above local mean
    filtered = []
    w = max(5, len(x)//30)
    for p in peaks:
        lo = max(0, p-w)
        hi = min(len(x), p+w)
        local = float(np.mean(x[lo:hi]))
        if x[p] - local >= min_prominence:
            filtered.append(p)

    # enforce spacing
    spaced = []
    min_gap = max(6, len(x)//20)
    for p in filtered:
        if not spaced or (p - spaced[-1]) >= min_gap:
            spaced.append(p)

    return len(spaced), spaced

def _detect_exercise(series: dict):
    """
    Heuristic exercise detection using which signal dominates.
    Returns one of: Squat, Lunge, Shoulder Abduction, Heel Raises, Unknown
    """
    knee = series["knee_angle_deg"]
    hipy = series["hip_y"]
    sh_y = series["shoulder_y"]

    knee_range = float(np.max(knee) - np.min(knee))
    hipy_range = float(np.max(hipy) - np.min(hipy))
    shy_range = float(np.max(sh_y) - np.min(sh_y))

    # Shoulder abduction often shows big shoulder/wrist movements,
    # but we only have shoulder_y proxy here; still can detect if upper body moves more than hip.
    if shy_range > hipy_range * 1.2 and knee_range < 25:
        return "Shoulder Abduction"

    # Heel raises: hip changes small, knee changes small, but ankle/hip y changes slightly.
    # We don't track ankle y directly in features; hipy_range small + knee_range small => likely heel raises.
    if hipy_range < 0.03 and knee_range < 20:
        return "Heel Raises"

    # Squat vs lunge: both have knee/hip changes; lunge sometimes has less symmetric movement.
    # We'll call it squat by default unless knee range is moderate and hip movement is smaller.
    if knee_range >= 25 and hipy_range >= 0.03:
        # If knee range is big -> squat/lunge.
        # This heuristic labels most lower-body bends as squat.
        return "Squat"

    return "Unknown"

def analyze_video(video_path: str, sample_fps: int = 10):
    ts, lms = _extract_landmarks_from_video(video_path, sample_fps=sample_fps)
    feat, series, summary = _compute_features(lms)

    exercise = _detect_exercise(series)

    # rep counting strategy
    # For squats/lunges: use hip_y peaks
    # For shoulder abduction: use shoulder_y peaks
    if exercise == "Shoulder Abduction":
        reps, peaks = _count_reps_from_signal(series["shoulder_y"], min_prominence=0.01)
    else:
        reps, peaks = _count_reps_from_signal(series["hip_y"], min_prominence=0.01)

    return {
        "timestamps_sec": ts,
        "feat_seq": feat,
        "series": series,
        "summary": summary,
        "exercise": exercise,
        "rep_count": int(reps),
        "rep_peak_frames": peaks,
    }
