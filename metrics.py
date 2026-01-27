import numpy as np

# --------------------------
# Basic geometry helpers
# --------------------------
def safe_norm(v: np.ndarray) -> float:
    n = float(np.linalg.norm(v))
    return n if n > 1e-9 else 1e-9

def angle_3pts(a, b, c) -> float:
    """
    Angle ABC in degrees where b is vertex.
    a,b,c: arrays shape (2,) or (3,)
    """
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    ba = a - b
    bc = c - b
    cosang = float(np.dot(ba, bc) / (safe_norm(ba) * safe_norm(bc)))
    cosang = max(-1.0, min(1.0, cosang))
    return float(np.degrees(np.arccos(cosang)))

def ema(x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    if len(x) == 0:
        return x
    y = np.zeros_like(x, dtype=np.float32)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y

# --------------------------
# Simple DTW for similarity
# --------------------------
def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    a: (T, D), b: (U, D)
    Returns normalized DTW distance.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    T, D = a.shape
    U, _ = b.shape

    # cost matrix
    INF = 1e9
    dp = np.full((T + 1, U + 1), INF, dtype=np.float32)
    dp[0, 0] = 0.0

    for i in range(1, T + 1):
        for j in range(1, U + 1):
            cost = float(np.linalg.norm(a[i - 1] - b[j - 1]))
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

    # Normalize by path length approx
    return float(dp[T, U] / (T + U))

def similarity_from_distance(d: float) -> float:
    """
    Convert a distance to a similarity score [0..1].
    """
    # This mapping is heuristic; good enough for demo.
    return float(np.exp(-2.0 * d))
