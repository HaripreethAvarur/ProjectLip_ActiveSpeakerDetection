"""
evaluate_asd.py  –  Active Speaker Detection Evaluation Suite
=============================================================
Generates quantitative metrics from a video file (no GPU required).

Modes
-----
1. --mode full        : Runs complete ASD pipeline on a video, compares against
                        auto-generated pseudo-GT (speech segments from VAD/energy),
                        outputs per-frame and summary metrics.
2. --mode benchmark   : Latency / FPS benchmarks for face detection + visual scoring.
3. --mode simulation  : Simulates ASD on a synthetic scenario (no video needed),
                        useful when no test video is available.

Metrics reported
----------------
  - Precision, Recall, F1 (active-speaker detection per frame)
  - Accuracy
  - Mean Average Precision (mAP) over score thresholds
  - FPS / latency percentiles (p50, p90, p99)
  - Face detection rate
  - Fusion score statistics
  - Confusion matrix

Usage examples
--------------
  python evaluate_asd.py --mode simulation --speakers 2 --frames 500
  python evaluate_asd.py --mode full --video myvideo.mp4 --no-audio
  python evaluate_asd.py --mode benchmark --frames 200
"""

import argparse
import collections
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="ASD Evaluation Suite")
    p.add_argument("--mode",      default="simulation",
                   choices=["full", "benchmark", "simulation"],
                   help="Evaluation mode (default: simulation)")
    p.add_argument("--video",     default=None,
                   help="Input video path (required for full/benchmark modes)")
    p.add_argument("--max-faces", type=int, default=2)
    p.add_argument("--min-score", type=float, default=0.35)
    p.add_argument("--frames",    type=int, default=300,
                   help="Max frames to process (full/benchmark) or simulate")
    p.add_argument("--speakers",  type=int, default=2,
                   help="Number of simulated speakers (simulation mode)")
    p.add_argument("--no-audio",  action="store_true",
                   help="Skip audio VAD (visual-only)")
    p.add_argument("--out",       default="./eval_results",
                   help="Output directory for metrics")
    p.add_argument("--device",    default="cpu")
    p.add_argument("--use-talknet", action="store_true",
                   help="Attempt to load TalkNet model (requires pretrain_TalkSet.model)")
    p.add_argument("--model",     default="pretrain_TalkSet.model")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, scores=None):
    """Binary classification metrics: Precision, Recall, F1, Accuracy, mAP."""
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    TP = int(np.sum((y_pred == 1) & (y_true == 1)))
    FP = int(np.sum((y_pred == 1) & (y_true == 0)))
    FN = int(np.sum((y_pred == 0) & (y_true == 1)))
    TN = int(np.sum((y_pred == 0) & (y_true == 0)))

    precision   = TP / max(TP + FP, 1)
    recall      = TP / max(TP + FN, 1)
    f1          = 2 * precision * recall / max(precision + recall, 1e-9)
    accuracy    = (TP + TN) / max(len(y_true), 1)
    specificity = TN / max(TN + FP, 1)

    # Mean Average Precision (area under PR curve via threshold sweep)
    map_score = 0.0
    if scores is not None:
        scores = np.array(scores, dtype=float)
        thresholds = np.linspace(0.0, 1.0, 51)
        prec_list, rec_list = [], []
        for t in thresholds:
            p_ = (scores >= t).astype(int)
            tp_ = int(np.sum((p_ == 1) & (y_true == 1)))
            fp_ = int(np.sum((p_ == 1) & (y_true == 0)))
            fn_ = int(np.sum((p_ == 0) & (y_true == 1)))
            pr = tp_ / max(tp_ + fp_, 1)
            rc = tp_ / max(tp_ + fn_, 1)
            prec_list.append(pr)
            rec_list.append(rc)
        # Area under the PR curve (trapezoidal)
        rec_arr  = np.array(rec_list)
        prec_arr = np.array(prec_list)
        sorted_idx = np.argsort(rec_arr)
        trapz_fn  = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
        map_score = float(trapz_fn(prec_arr[sorted_idx], rec_arr[sorted_idx]))

    return {
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "Precision":    round(precision, 4),
        "Recall":       round(recall,    4),
        "F1":           round(f1,        4),
        "Accuracy":     round(accuracy,  4),
        "Specificity":  round(specificity, 4),
        "mAP":          round(map_score, 4),
    }


def latency_stats(ms_list):
    a = np.array(ms_list, dtype=float)
    return {
        "mean_ms":  round(float(np.mean(a)),              2),
        "std_ms":   round(float(np.std(a)),               2),
        "p50_ms":   round(float(np.percentile(a, 50)),    2),
        "p90_ms":   round(float(np.percentile(a, 90)),    2),
        "p99_ms":   round(float(np.percentile(a, 99)),    2),
        "min_ms":   round(float(np.min(a)),               2),
        "max_ms":   round(float(np.max(a)),               2),
        "mean_fps": round(float(1000.0 / max(np.mean(a), 1e-6)), 1),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Face detection helper
# ──────────────────────────────────────────────────────────────────────────────

def get_face_cascade():
    import os
    cascade_path = None
    if hasattr(cv2, "data"):
        cascade_path = os.path.join(cv2.data.haarcascades,
                                    "haarcascade_frontalface_default.xml")
    if cascade_path is None or not os.path.isfile(cascade_path):
        cv2_root = os.path.dirname(cv2.__file__)
        candidates = [
            os.path.join(cv2_root, "data", "haarcascade_frontalface_default.xml"),
            os.path.join(cv2_root, "../share/opencv4/haarcascades",
                         "haarcascade_frontalface_default.xml"),
        ]
        for c in candidates:
            c = os.path.normpath(c)
            if os.path.isfile(c):
                cascade_path = c
                break
    if cascade_path is None:
        raise FileNotFoundError("haarcascade_frontalface_default.xml not found. "
                                "Install opencv-python.")
    return cv2.CascadeClassifier(cascade_path)


def detect_faces(frame, cascade, max_faces=2):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dets = cascade.detectMultiScale(gray, scaleFactor=1.1,
                                    minNeighbors=5, minSize=(60, 60))
    boxes = []
    if len(dets) > 0:
        for (x, y, w, h) in sorted(dets, key=lambda d: d[0])[:max_faces]:
            boxes.append((x, y, x + w, y + h))
    return boxes


# ──────────────────────────────────────────────────────────────────────────────
# Lightweight visual-only scorer (no TalkNet needed)
# ──────────────────────────────────────────────────────────────────────────────

class LightweightVisualScorer:
    """
    CPU-only visual activity scorer using optical flow + mouth-region variance.
    Provides a continuous [0,1] speaking score without TalkNet.
    Suitable for CPU-only machines (your teammate's environment).
    """
    def __init__(self, window=11, smoothing=8):
        self.window    = window
        self.smoothing = smoothing
        self._prev_frames:  dict = {}
        self._score_history: dict = {}

    def score(self, face_id: int, face_bgr) -> float:
        if face_bgr is None or face_bgr.size == 0:
            return 0.0

        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64)).astype(np.float32) / 255.0

        if face_id not in self._score_history:
            self._score_history[face_id] = collections.deque(maxlen=self.smoothing)
            self._prev_frames[face_id]   = gray
            return 0.0

        prev = self._prev_frames[face_id]

        # --- Optical flow magnitude in mouth region (bottom 40% of face crop)
        h = gray.shape[0]
        mouth_curr = gray[int(h*0.55):, :]
        mouth_prev = prev[int(h*0.55):, :]
        diff = np.abs(mouth_curr.astype(float) - mouth_prev.astype(float))
        flow_score = float(np.clip(diff.mean() * 20.0, 0.0, 1.0))

        # --- Laplacian variance (sharpness) as proxy for mouth motion
        mouth_u8 = (mouth_curr * 255.0).clip(0, 255).astype(np.uint8)
        lap_var = float(cv2.Laplacian(mouth_u8, cv2.CV_64F).var())
        sharp_score = float(np.clip(lap_var / 50.0, 0.0, 1.0))

        raw = 0.7 * flow_score + 0.3 * sharp_score
        self._prev_frames[face_id] = gray
        self._score_history[face_id].append(raw)
        return float(np.mean(self._score_history[face_id]))


# ──────────────────────────────────────────────────────────────────────────────
# Energy-based pseudo ground truth
# ──────────────────────────────────────────────────────────────────────────────

def extract_energy_gt(video_path, n_frames):
    """
    Extract per-frame energy from video audio track as pseudo ground truth.
    Returns binary array (1 = speech, 0 = silence).
    """
    try:
        import subprocess, tempfile, wave, struct
        tmp_wav = tempfile.mktemp(suffix=".wav")
        cmd = ["ffmpeg", "-i", video_path, "-ar", "16000", "-ac", "1",
               "-y", tmp_wav, "-loglevel", "quiet"]
        subprocess.run(cmd, check=True, timeout=30)

        with wave.open(tmp_wav, "rb") as wf:
            sr       = wf.getframerate()
            n_audio  = wf.getnframes()
            raw      = wf.readframes(n_audio)

        samples = np.frombuffer(raw, dtype=np.int16).astype(float) / 32768.0
        # Resample to frame-level: assume 25fps default
        fps_est   = 25
        hop       = sr // fps_est
        energies  = []
        for i in range(n_frames):
            start = i * hop
            end   = start + hop
            chunk = samples[start:end] if end <= len(samples) else np.zeros(hop)
            energies.append(float(np.sqrt(np.mean(chunk**2))))

        Path(tmp_wav).unlink(missing_ok=True)
        energies  = np.array(energies)
        threshold = np.percentile(energies, 40)   # bottom 40% = silence
        return (energies > threshold).astype(int)

    except Exception as e:
        print(f"[GT] Audio extraction failed ({e}), using synthetic GT")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# MODE: SIMULATION
# ──────────────────────────────────────────────────────────────────────────────

def run_simulation(args, out_dir):
    """
    Simulate ASD on synthetic data. No video or model needed.
    Generates realistic score distributions and computes all metrics.
    """
    print("\n[Simulation] Generating synthetic ASD evaluation...")
    rng       = np.random.default_rng(42)
    n_frames  = args.frames
    n_sp      = args.speakers

    # Ground truth: one speaker speaks at a time, switches every ~40 frames
    gt_speaker = np.zeros(n_frames, dtype=int)  # -1 = silence
    cur_sp, seg_len = 0, 0
    for f in range(n_frames):
        seg_len += 1
        if seg_len > rng.integers(30, 70):
            # 20% chance of silence
            cur_sp = -1 if rng.random() < 0.2 else (cur_sp + 1) % n_sp
            seg_len = 0
        gt_speaker[f] = cur_sp

    # Simulate scores per speaker per frame
    speaker_scores = {}
    for sp in range(n_sp):
        scores = np.zeros(n_frames)
        for f in range(n_frames):
            is_active = (gt_speaker[f] == sp)
            base  = rng.normal(0.72, 0.12) if is_active else rng.normal(0.22, 0.10)
            scores[f] = float(np.clip(base, 0.0, 1.0))
        speaker_scores[sp] = scores

    # Fusion: pick highest scoring speaker if above threshold
    pred_speaker = np.full(n_frames, -1, dtype=int)
    for f in range(n_frames):
        best_sp    = max(range(n_sp), key=lambda s: speaker_scores[s][f])
        best_score = speaker_scores[best_sp][f]
        if best_score >= args.min_score:
            pred_speaker[f] = best_sp

    # Per-frame binary: was ANY speaker correctly identified?
    y_true, y_pred, y_scores = [], [], []
    for f in range(n_frames):
        gt_active   = int(gt_speaker[f] != -1)
        pred_active = int(pred_speaker[f] != -1)
        best_score  = max(speaker_scores[s][f] for s in range(n_sp))
        y_true.append(gt_active)
        y_pred.append(pred_active)
        y_scores.append(best_score)

    # Per-speaker identity accuracy (when someone IS predicted active)
    identity_correct = 0
    identity_total   = 0
    for f in range(n_frames):
        if pred_speaker[f] != -1 and gt_speaker[f] != -1:
            identity_total += 1
            if pred_speaker[f] == gt_speaker[f]:
                identity_correct += 1

    identity_acc = identity_correct / max(identity_total, 1)

    # Latency: simulate realistic CPU inference times
    latencies = list(rng.normal(18.5, 4.2, n_frames).clip(8.0, 60.0))

    metrics = compute_metrics(y_true, y_pred, y_scores)
    lat     = latency_stats(latencies)

    # Fusion score stats
    all_scores = np.concatenate([speaker_scores[s] for s in range(n_sp)])

    results = {
        "mode": "simulation",
        "config": {
            "n_frames": n_frames,
            "n_speakers": n_sp,
            "min_score_threshold": args.min_score,
        },
        "detection_metrics": metrics,
        "identity_accuracy": round(identity_acc, 4),
        "latency_ms": lat,
        "score_stats": {
            "mean":   round(float(all_scores.mean()), 4),
            "std":    round(float(all_scores.std()),  4),
            "min":    round(float(all_scores.min()),  4),
            "max":    round(float(all_scores.max()),  4),
        },
        "speaking_ratio": {
            f"speaker_{s}": round(float((gt_speaker == s).mean()), 4)
            for s in range(n_sp)
        },
    }

    return results, y_true, y_pred, y_scores


# ──────────────────────────────────────────────────────────────────────────────
# MODE: BENCHMARK
# ──────────────────────────────────────────────────────────────────────────────

def run_benchmark(args, out_dir):
    """Benchmark face detection + visual scoring speed."""
    print("\n[Benchmark] Running latency benchmark...")

    # Check video
    if not args.video:
        print("[Benchmark] No --video provided; using synthetic frames.")
        use_synthetic = True
    else:
        cap = cv2.VideoCapture(args.video)
        use_synthetic = not cap.isOpened()
        if use_synthetic:
            print("[Benchmark] Cannot open video; using synthetic frames.")
        else:
            cap.release()

    cascade = get_face_cascade()
    scorer  = LightweightVisualScorer()

    det_latencies   = []
    score_latencies = []
    face_counts     = []
    n               = args.frames

    def get_frame(idx):
        if use_synthetic:
            f = np.random.randint(0, 200, (480, 640, 3), dtype=np.uint8)
            # draw synthetic face-like blobs
            cx, cy = 200 + (idx % 4) * 10, 200
            cv2.ellipse(f, (cx, cy), (60, 80), 0, 0, 360, (200, 180, 160), -1)
            return f
        return None

    if not use_synthetic:
        cap = cv2.VideoCapture(args.video)

    for i in range(n):
        if use_synthetic:
            frame = get_frame(i)
        else:
            ret, frame = cap.read()
            if not ret:
                break

        t0 = time.perf_counter()
        boxes = detect_faces(frame, cascade, args.max_faces)
        det_latencies.append((time.perf_counter() - t0) * 1000)
        face_counts.append(len(boxes))

        t1 = time.perf_counter()
        for fid, (x1, y1, x2, y2) in enumerate(boxes):
            crop = frame[max(0,y1):y2, max(0,x1):x2]
            scorer.score(fid, crop)
        score_latencies.append((time.perf_counter() - t1) * 1000)

    if not use_synthetic:
        cap.release()

    total_lat = [d + s for d, s in zip(det_latencies, score_latencies)]

    results = {
        "mode": "benchmark",
        "n_frames": len(det_latencies),
        "face_detection": latency_stats(det_latencies),
        "visual_scoring": latency_stats(score_latencies),
        "total_pipeline": latency_stats(total_lat),
        "face_count_stats": {
            "mean_faces_per_frame": round(float(np.mean(face_counts)), 2),
            "face_detection_rate":  round(float(np.mean([c > 0 for c in face_counts])), 4),
        },
    }
    return results


# ──────────────────────────────────────────────────────────────────────────────
# MODE: FULL (video inference + metric computation)
# ──────────────────────────────────────────────────────────────────────────────

def run_full(args, out_dir):
    """Full inference on a video with metric computation."""
    if not args.video or not Path(args.video).exists():
        print("[Full] No valid --video provided. Falling back to simulation.")
        return run_simulation(args, out_dir)[0]

    print(f"\n[Full] Processing: {args.video}")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("[Full] Cannot open video. Falling back to simulation.")
        return run_simulation(args, out_dir)[0]

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames           = min(args.frames, total_video_frames)
    print(f"[Full] {total_video_frames} frames @ {fps_video:.1f}fps. Processing {n_frames}.")

    cascade = get_face_cascade()
    scorer  = LightweightVisualScorer()

    # Try TalkNet if requested
    talknet_streamer = None
    if args.use_talknet and Path(args.model).exists():
        try:
            sys.path.insert(0, str(Path(__file__).parent / "TalkNet-ASD"))
            from talknet_inference import TalkNetStreamer, load_talknet
            talknet_model    = load_talknet(args.model, args.device)
            talknet_streamer = TalkNetStreamer(talknet_model, device=args.device)
            print("[Full] TalkNet loaded.")
        except Exception as e:
            print(f"[Full] TalkNet unavailable ({e}). Using lightweight scorer.")

    frame_records = []
    latencies     = []
    face_counts   = []

    for fi in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        t0    = time.perf_counter()
        boxes = detect_faces(frame, cascade, args.max_faces)
        face_counts.append(len(boxes))

        frame_scores = {}
        for fid, (x1, y1, x2, y2) in enumerate(boxes):
            pad = 0.15
            H, W = frame.shape[:2]
            pw = int((x2-x1)*pad); ph = int((y2-y1)*pad)
            crop = frame[max(0,y1-ph):min(H,y2+ph), max(0,x1-pw):min(W,x2+pw)]
            if talknet_streamer:
                vs = talknet_streamer.update_video(fid, crop)
            else:
                vs = scorer.score(fid, crop)
            frame_scores[fid] = vs

        # Pick active speaker
        active_sp = None
        if frame_scores:
            best = max(frame_scores, key=frame_scores.__getitem__)
            if frame_scores[best] >= args.min_score:
                active_sp = best

        latencies.append((time.perf_counter() - t0) * 1000)
        frame_records.append({
            "frame": fi,
            "n_faces": len(boxes),
            "active_speaker": active_sp,
            "scores": frame_scores,
            "max_score": max(frame_scores.values()) if frame_scores else 0.0,
        })

    cap.release()

    # Pseudo GT from audio energy
    gt_binary = extract_energy_gt(args.video, len(frame_records))
    if gt_binary is None:
        # fallback: assume active when any face detected
        gt_binary = np.array([(1 if r["n_faces"] > 0 else 0)
                               for r in frame_records])

    y_pred   = [1 if r["active_speaker"] is not None else 0 for r in frame_records]
    y_scores = [r["max_score"] for r in frame_records]
    y_true   = gt_binary[:len(y_pred)].tolist()

    metrics = compute_metrics(y_true, y_pred, y_scores)
    lat     = latency_stats(latencies)

    all_scores = [r["max_score"] for r in frame_records]

    results = {
        "mode": "full",
        "video": str(args.video),
        "n_frames_processed": len(frame_records),
        "fps_video": fps_video,
        "scorer": "TalkNet" if talknet_streamer else "LightweightVisual",
        "detection_metrics": metrics,
        "latency_ms": lat,
        "face_stats": {
            "mean_faces_per_frame": round(float(np.mean(face_counts)), 3),
            "face_detection_rate":  round(float(np.mean([c > 0 for c in face_counts])), 4),
        },
        "score_stats": {
            "mean":   round(float(np.mean(all_scores)),  4),
            "std":    round(float(np.std(all_scores)),   4),
            "min":    round(float(np.min(all_scores)),   4),
            "max":    round(float(np.max(all_scores)),   4),
        },
    }
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Report printing
# ──────────────────────────────────────────────────────────────────────────────

def print_report(results):
    SEP = "=" * 62

    print(f"\n{SEP}")
    print(f"  ASD EVALUATION REPORT  —  mode: {results['mode'].upper()}")
    print(SEP)

    if "config" in results:
        cfg = results["config"]
        print(f"\n  Config:")
        for k, v in cfg.items():
            print(f"    {k:<30} {v}")

    dm = results.get("detection_metrics", {})
    if dm:
        print(f"\n  Detection Metrics  (per-frame binary active-speaker)")
        print(f"    {'Metric':<30} {'Value':>10}")
        print(f"    {'-'*42}")
        for key in ["Precision","Recall","F1","Accuracy","Specificity","mAP"]:
            print(f"    {key:<30} {dm.get(key,0):>10.4f}")
        print(f"\n  Confusion Matrix")
        print(f"    TP={dm.get('TP',0)}  FP={dm.get('FP',0)}  FN={dm.get('FN',0)}  TN={dm.get('TN',0)}")

    if "identity_accuracy" in results:
        print(f"\n  Speaker Identity Accuracy    {results['identity_accuracy']:.4f}")

    lat = results.get("latency_ms") or results.get("total_pipeline", {})
    if lat:
        print(f"\n  Inference Latency")
        print(f"    {'Metric':<30} {'Value':>10}")
        print(f"    {'-'*42}")
        for key in ["mean_ms","std_ms","p50_ms","p90_ms","p99_ms","mean_fps"]:
            print(f"    {key:<30} {lat.get(key,0):>10.2f}")

    ss = results.get("score_stats", {})
    if ss:
        print(f"\n  Fusion/Visual Score Distribution")
        for k, v in ss.items():
            print(f"    {k:<30} {v:>10.4f}")

    fs = results.get("face_stats") or results.get("face_count_stats", {})
    if fs:
        print(f"\n  Face Detection")
        for k, v in fs.items():
            print(f"    {k:<30} {v:>10.4f}")

    if "face_detection" in results:
        print(f"\n  Face Detection Latency")
        fd = results["face_detection"]
        for k in ["mean_ms","p90_ms","mean_fps"]:
            print(f"    {k:<30} {fd.get(k,0):>10.2f}")
        print(f"\n  Visual Scoring Latency")
        vs = results["visual_scoring"]
        for k in ["mean_ms","p90_ms","mean_fps"]:
            print(f"    {k:<30} {vs.get(k,0):>10.2f}")

    print(f"\n{SEP}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Save results
# ──────────────────────────────────────────────────────────────────────────────

def save_results(results, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    mode = results["mode"]

    # JSON
    json_path = Path(out_dir) / f"eval_{mode}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Saved] {json_path}")

    # Flat CSV summary
    csv_path = Path(out_dir) / f"eval_{mode}_summary.csv"
    flat = {}
    for k, v in results.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                if not isinstance(vv, dict):
                    flat[f"{k}.{kk}"] = vv
        elif not isinstance(v, (dict, list)):
            flat[k] = v

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in flat.items():
            writer.writerow([k, v])
    print(f"[Saved] {csv_path}")

    # Human-readable report
    report_path = Path(out_dir) / f"eval_{mode}_report.txt"
    dm  = results.get("detection_metrics", {})
    lat = results.get("latency_ms") or results.get("total_pipeline", {})

    with open(report_path, "w") as f:
        f.write("ASD Evaluation Report\n")
        f.write("=" * 62 + "\n\n")
        f.write(f"Mode   : {results['mode']}\n")
        if "video" in results:
            f.write(f"Video  : {results['video']}\n")
        if "scorer" in results:
            f.write(f"Scorer : {results['scorer']}\n")
        f.write("\nDetection Metrics\n")
        f.write("-" * 40 + "\n")
        for key in ["Precision","Recall","F1","Accuracy","Specificity","mAP"]:
            f.write(f"  {key:<28} {dm.get(key, 0):.4f}\n")
        f.write(f"\n  Confusion matrix: TP={dm.get('TP',0)} FP={dm.get('FP',0)} "
                f"FN={dm.get('FN',0)} TN={dm.get('TN',0)}\n")
        if "identity_accuracy" in results:
            f.write(f"\n  Speaker Identity Accuracy: {results['identity_accuracy']:.4f}\n")
        if lat:
            f.write("\nLatency\n")
            f.write("-" * 40 + "\n")
            for key in ["mean_ms","p50_ms","p90_ms","p99_ms","mean_fps"]:
                f.write(f"  {key:<28} {lat.get(key,0):.2f}\n")
        f.write("\n" + "=" * 62 + "\n")
    print(f"[Saved] {report_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    out_dir = args.out

    if args.mode == "simulation":
        results, y_true, y_pred, y_scores = run_simulation(args, out_dir)
    elif args.mode == "benchmark":
        results = run_benchmark(args, out_dir)
    elif args.mode == "full":
        results = run_full(args, out_dir)
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)

    print_report(results)
    save_results(results, out_dir)


if __name__ == "__main__":
    main()
