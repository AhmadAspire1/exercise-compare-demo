import os
import json
import tempfile
import numpy as np
import streamlit as st

from pose_analyzer import analyze_video
from metrics import dtw_distance, similarity_from_distance
from feedback import generate_feedback

st.set_page_config(page_title="Exercise Compare Demo", layout="wide")

st.title("Exercise Compare Demo (Tutorial vs Patient)")
st.caption("Local demo: upload two videos → detect exercise, reps, form metrics, similarity score.")

with st.sidebar:
    st.header("Settings")
    sample_fps = st.slider("Pose sampling FPS (higher = slower)", 5, 20, 10, 1)
    st.markdown("---")
    st.write("Tips for best results:")
    st.write("- Full body visible (head to feet)")
    st.write("- Stable camera (no heavy shake)")
    st.write("- Good lighting")
    st.write("- Similar camera angle for both videos")

col1, col2 = st.columns(2)
with col1:
    tut_file = st.file_uploader("Upload tutorial video (gold standard)", type=["mp4", "mov", "mkv", "avi"], key="tut")
with col2:
    pat_file = st.file_uploader("Upload patient attempt video", type=["mp4", "mov", "mkv", "avi"], key="pat")

analyze = st.button("Analyze", type="primary", disabled=not (tut_file and pat_file))

def save_uploaded_to_temp(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

def compare_summaries(tut, pat):
    # DTW on feature sequences
    d = dtw_distance(tut["feat_seq"], pat["feat_seq"])
    sim = similarity_from_distance(d)

    # Basic metric deltas
    tut_sum = tut["summary"]
    pat_sum = pat["summary"]

    deltas = {
        "knee_angle_mean_deg_delta": pat_sum["knee_angle_mean_deg"] - tut_sum["knee_angle_mean_deg"],
        "knee_angle_min_deg_delta": pat_sum["knee_angle_min_deg"] - tut_sum["knee_angle_min_deg"],
        "trunk_lean_mean_delta": pat_sum["trunk_lean_mean"] - tut_sum["trunk_lean_mean"],
        "valgus_mean_delta": pat_sum["valgus_mean"] - tut_sum["valgus_mean"],
        "depth_proxy_delta": pat_sum["depth_proxy"] - tut_sum["depth_proxy"],
    }

    # Simple flags (demo-level)
    flags = []
    if deltas["depth_proxy_delta"] < -10:
        flags.append("Depth is noticeably shallower than the tutorial (proxy).")
    if deltas["trunk_lean_mean_delta"] > 0.02:
        flags.append("More forward trunk lean than the tutorial (proxy).")
    if deltas["valgus_mean_delta"] > 0.02:
        flags.append("More knee-in / valgus tendency than the tutorial (proxy).")
    if abs(pat["rep_count"] - tut["rep_count"]) >= 2:
        flags.append("Rep count differs a lot (could be missed detection or different pacing).")

    return d, sim, deltas, flags

if analyze:
    try:
        tut_path = save_uploaded_to_temp(tut_file)
        pat_path = save_uploaded_to_temp(pat_file)

        with st.spinner("Analyzing tutorial video..."):
            tut = analyze_video(tut_path, sample_fps=sample_fps)

        with st.spinner("Analyzing patient video..."):
            pat = analyze_video(pat_path, sample_fps=sample_fps)

        # Compare
        d, sim, deltas, flags = compare_summaries(tut, pat)

        # UI
        st.markdown("---")
        left, right = st.columns([1, 1])

        with left:
            st.subheader("Tutorial (Gold Standard)")
            st.write(f"**Detected exercise:** {tut['exercise']}")
            st.write(f"**Estimated reps:** {tut['rep_count']}")
            st.write("**Summary metrics**")
            st.json(tut["summary"])

        with right:
            st.subheader("Patient Attempt")
            st.write(f"**Detected exercise:** {pat['exercise']}")
            st.write(f"**Estimated reps:** {pat['rep_count']}")
            st.write("**Summary metrics**")
            st.json(pat["summary"])

        st.markdown("## Comparison Result")
        c1, c2, c3 = st.columns(3)
        c1.metric("Similarity (0..1)", f"{sim:.3f}")
        c2.metric("DTW Distance (lower = better)", f"{d:.4f}")
        c3.metric("Exercise Match", "Yes" if tut["exercise"] == pat["exercise"] and tut["exercise"] != "Unknown" else "Maybe")

        st.write("**Metric differences (Patient - Tutorial):**")
        st.json(deltas)

        st.markdown("## Coaching Feedback")

        tips = generate_feedback(tut, pat, deltas)

        for tip in tips:
         st.info(tip)

        # Simple time-series display
        st.markdown("## Signals (for debugging / transparency)")
        sig_col1, sig_col2 = st.columns(2)

        with sig_col1:
            st.write("Knee angle (deg) - tutorial vs patient")
            st.line_chart({
                "tutorial_knee_deg": tut["series"]["knee_angle_deg"],
                "patient_knee_deg": pat["series"]["knee_angle_deg"],
            })

        with sig_col2:
            st.write("Hip vertical (proxy) - tutorial vs patient")
            st.line_chart({
                "tutorial_hip_y": tut["series"]["hip_y"],
                "patient_hip_y": pat["series"]["hip_y"],
            })

        # Download JSON report
        report = {
            "tutorial": {
                "exercise": tut["exercise"],
                "rep_count": tut["rep_count"],
                "summary": tut["summary"],
            },
            "patient": {
                "exercise": pat["exercise"],
                "rep_count": pat["rep_count"],
                "summary": pat["summary"],
            },
            "comparison": {
                "dtw_distance": d,
                "similarity": sim,
                "deltas_patient_minus_tutorial": deltas,
                "feedback_flags": flags,
            }
        }
        st.download_button(
            "Download JSON report",
            data=json.dumps(report, indent=2),
            file_name="exercise_comparison_report.json",
            mime="application/json"
        )

    except Exception as e:
        st.error(str(e))
    finally:
        # Cleanup temp files
        try:
            if 'tut_path' in locals() and os.path.exists(tut_path):
                os.remove(tut_path)
            if 'pat_path' in locals() and os.path.exists(pat_path):
                os.remove(pat_path)
        except Exception:
            pass
