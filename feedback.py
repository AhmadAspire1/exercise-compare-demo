def generate_feedback(tut, pat, deltas):
    """
    Generate patient-friendly coaching feedback
    based on metric deltas (patient - tutorial).
    """
    feedback = []

    exercise = tut["exercise"]

    # ---- Depth (mainly for squat/lunge) ----
    if exercise in ["Squat", "Lunge"]:
        if deltas["depth_proxy_delta"] < -10:
            feedback.append(
                "Try to go deeper during each rep. Bend your knees more at the bottom "
                "while keeping your heels on the ground."
            )

    # ---- Trunk lean ----
    if deltas["trunk_lean_mean_delta"] > 0.02:
        feedback.append(
            "Keep your chest more upright. Think about pushing your chest slightly up "
            "and sitting straight down instead of leaning forward."
        )

    # ---- Knee valgus ----
    if deltas["valgus_mean_delta"] > 0.02:
        feedback.append(
            "Avoid letting your knees collapse inward. "
            "Focus on gently pushing your knees outward as you move down and up."
        )

    # ---- Rep mismatch ----
    if abs(pat["rep_count"] - tut["rep_count"]) >= 2:
        feedback.append(
            "Try to match the number of repetitions and pacing shown in the tutorial."
        )

    # ---- Generic tempo feedback ----
    feedback.append(
        "Move in a controlled way: slow down the descent and avoid bouncing at the bottom."
    )

    # Fallback
    if not feedback:
        feedback.append(
            "Your form is very close to the tutorial. Keep practicing with the same technique."
        )

    return feedback
