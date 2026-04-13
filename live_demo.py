import argparse
import collections
import csv
import sys
import time

import cv2
import numpy as np
import torch

from talknet_inference import TalkNetStreamer, load_talknet
from audio_modules import VoiceActivityDetector, WhisperTranscriber
from fusion import AudioVisualFusion

FONT = cv2.FONT_HERSHEY_SIMPLEX
SPEAKER_COLORS = {0: (0, 210, 0), 1: (0, 140, 255), 2: (255, 60, 120)}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera",     type=int,   default=0)
    parser.add_argument("--model",      default="pretrain_TalkSet.model")
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--mic",        type=int,   default=None)
    parser.add_argument("--whisper",    default="tiny")
    parser.add_argument("--chunk",      type=float, default=3.0)
    parser.add_argument("--max-faces",  type=int,   default=2)
    parser.add_argument("--no-audio",   action="store_true")
    parser.add_argument("--list-mics",  action="store_true")
    parser.add_argument("--log",        default=None)
    parser.add_argument("--min-score",  type=float, default=0.35)
    parser.add_argument("--threads",    type=int,   default=None)
    return parser.parse_args()


def detect_faces(frame, cascade, max_faces):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = cascade.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors=5, minSize=(60, 60))
    boxes = []
    if len(detections) > 0:
        for (x, y, w, h) in sorted(detections, key=lambda d: d[0])[:max_faces]:
            boxes.append((x, y, x + w, y + h))
    return boxes


def crop_face_with_padding(frame, x1, y1, x2, y2, padding=0.25):
    h, w = frame.shape[:2]
    pad_w = int((x2 - x1) * padding)
    pad_h = int((y2 - y1) * padding)
    return frame[max(0, y1-pad_h):min(h, y2+pad_h),
                 max(0, x1-pad_w):min(w, x2+pad_w)]


class FaceTracker:
    def __init__(self, max_faces=2, distance_threshold=0.30):
        self.max_faces = max_faces
        self.distance_threshold = distance_threshold
        self._tracked_faces = {}
        self._next_id = 0

    def update(self, face_centroids):
        if not face_centroids:
            return []
        if not self._tracked_faces:
            assigned_ids = []
            for centroid in face_centroids[:self.max_faces]:
                self._tracked_faces[self._next_id] = centroid
                assigned_ids.append(self._next_id)
                self._next_id += 1
            return assigned_ids

        assigned_ids = []
        used_ids = set()
        for centroid in face_centroids[:self.max_faces]:
            closest_id, closest_dist = None, float("inf")
            for face_id, known_centroid in self._tracked_faces.items():
                if face_id in used_ids:
                    continue
                dist = np.hypot(centroid[0] - known_centroid[0],
                                centroid[1] - known_centroid[1])
                if dist < closest_dist:
                    closest_dist, closest_id = dist, face_id

            if closest_id is not None and closest_dist < self.distance_threshold:
                self._tracked_faces[closest_id] = centroid
                used_ids.add(closest_id)
                assigned_ids.append(closest_id)
            else:
                self._tracked_faces[self._next_id] = centroid
                assigned_ids.append(self._next_id)
                used_ids.add(self._next_id)
                self._next_id += 1

        return assigned_ids


def draw_speaker_box(frame, x1, y1, x2, y2, color, label, fused_score, is_speaking):
    if is_speaking:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        corner_length = max(12, (x2 - x1) // 6)
        for cx, cy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(frame, (cx, cy), (cx + dx*corner_length, cy), color, 4)
            cv2.line(frame, (cx, cy), (cx, cy + dy*corner_length), color, 4)
        tag = f"{label}  SPEAKING"
        (tw, th), _ = cv2.getTextSize(tag, FONT, 0.62, 1)
        cv2.rectangle(frame, (x1, y1-th-14), (x1+tw+10, y1), color, -1)
        cv2.putText(frame, tag, (x1+5, y1-6), FONT, 0.62, (0, 0, 0), 2, cv2.LINE_AA)
    else:
        dim_color = tuple(int(c * 0.4) for c in color)
        cv2.rectangle(frame, (x1, y1), (x2, y2), dim_color, 1)
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.50, 1)
        cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+6, y1), dim_color, -1)
        cv2.putText(frame, label, (x1+3, y1-4), FONT, 0.50, (180, 180, 180), 1, cv2.LINE_AA)


def draw_score_bars(frame, x, y, visual_score, audio_vad, fused_score, frame_height):
    if y + 56 >= frame_height:
        return
    bar_data = [
        ("vis", visual_score, (200, 150, 255)),
        ("aud", audio_vad, (100, 220, 190)),
        ("fus", fused_score, (255, 220, 80)),
    ]
    for label, value, color in bar_data:
        cv2.rectangle(frame, (x, y), (x+80, y+9), (50, 50, 50), -1)
        cv2.rectangle(frame, (x, y), (x+int(80*value), y+9), color, -1)
        cv2.putText(frame, f"{label}:{value:.2f}", (x, y-2),
                    FONT, 0.36, (190, 190, 190), 1, cv2.LINE_AA)
        y += 15


def draw_caption_bar(frame, transcript):
    if not transcript:
        return
    h, w = frame.shape[:2]
    display_text = ("… " + transcript[-90:]) if len(transcript) > 90 else transcript
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-52), (w, h), (8, 8, 8), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (0, h-52), (7, h), (180, 180, 180), -1)
    cv2.putText(frame, display_text, (14, h-16), FONT, 0.62, (255, 255, 255), 1, cv2.LINE_AA)


def draw_hud(frame, fps, frame_ms, num_faces, vad_prob):
    info = f"FPS:{fps:.1f}  {frame_ms:.0f}ms  faces:{num_faces}  vad:{vad_prob:.2f}"
    cv2.putText(frame, info, (8, 24), FONT, 0.46, (160, 160, 160), 1, cv2.LINE_AA)


class PerformanceLogger:
    def __init__(self, log_path):
        self.log_path = log_path
        self._records = []

    def log(self, frame_idx, fps, frame_ms, vad_prob, num_faces, active_speaker_id):
        self._records.append({
            "frame": frame_idx,
            "fps": round(fps, 1),
            "ms": round(frame_ms, 2),
            "vad": round(vad_prob, 3),
            "faces": num_faces,
            "active_speaker": active_speaker_id,
        })

    def save(self):
        if not self._records:
            return
        from pathlib import Path
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._records[0].keys())
            writer.writeheader()
            writer.writerows(self._records)
        print(f"[Perf] {len(self._records)} frames logged → {self.log_path}")


def main():
    args = parse_args()

    if args.list_mics:
        import sounddevice as sd
        print(sd.query_devices())
        sys.exit(0)

    if args.threads:
        torch.set_num_threads(args.threads)

    print("[System] Loading TalkNet...")
    talknet_model = load_talknet(args.model, args.device)
    streamer = TalkNetStreamer(talknet_model, device=args.device)
    fusion = AudioVisualFusion()

    vad = None
    if not args.no_audio:
        try:
            vad = VoiceActivityDetector(mic_device=args.mic)
            vad.add_audio_callback(streamer.update_audio)
            vad.start()
        except Exception as e:
            print(f"[VAD] Could not start: {e}")

    transcriber = None
    if not args.no_audio:
        try:
            transcriber = WhisperTranscriber(args.whisper, args.chunk, args.mic)
            transcriber.start()
        except Exception as e:
            print(f"[Whisper] Could not start: {e}")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    face_tracker = FaceTracker(max_faces=args.max_faces)
    perf_logger = PerformanceLogger(args.log) if args.log else None

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[Error] Camera {args.camera} not found")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Camera] {frame_width}×{frame_height}   Q to quit")

    prev_time = time.perf_counter()
    fps_history = collections.deque(maxlen=30)
    last_speaking_frame = {}
    persist_frames = 30
    frame_idx = 0

    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        face_boxes = detect_faces(frame, face_cascade, args.max_faces)
        face_centroids = [
            ((x1+x2)/2/frame_width, (y1+y2)/2/frame_height)
            for x1, y1, x2, y2 in face_boxes
        ]
        face_ids = face_tracker.update(face_centroids)
        audio_vad = vad.get_speech_probability() if vad else 0.5

        visual_scores = {}
        for box, face_id in zip(face_boxes, face_ids):
            x1, y1, x2, y2 = box
            face_crop = crop_face_with_padding(frame, x1, y1, x2, y2)
            if face_crop.size > 0:
                visual_scores[face_id] = streamer.update_video(face_id, face_crop)

        active_speaker = fusion.get_active_speaker(
            face_ids, visual_scores, audio_vad, args.min_score)

        if active_speaker is not None:
            last_speaking_frame[active_speaker] = frame_idx

        for box, face_id in zip(face_boxes, face_ids):
            x1, y1, x2, y2 = box
            fused_score = fusion.get_fused_score(face_id)
            frames_since_speaking = frame_idx - last_speaking_frame.get(face_id, -9999)
            is_speaking = (face_id == active_speaker) or (frames_since_speaking <= persist_frames)
            color = SPEAKER_COLORS.get(face_id % len(SPEAKER_COLORS), SPEAKER_COLORS[0])
            draw_speaker_box(frame, x1, y1, x2, y2, color,
                             f"Speaker {face_id+1}", fused_score, is_speaking)
            draw_score_bars(frame, x1, y2+4,
                            visual_scores.get(face_id, 0.0),
                            audio_vad, fused_score, frame_height)

        transcript = transcriber.get_latest() if transcriber else ""
        draw_caption_bar(frame, transcript)

        now = time.perf_counter()
        fps_history.append(1.0 / max(now - prev_time, 1e-6))
        prev_time = now
        fps = float(np.mean(fps_history))
        frame_ms = (now - t0) * 1000
        draw_hud(frame, fps, frame_ms, len(face_boxes), audio_vad)

        if perf_logger:
            perf_logger.log(frame_idx, fps, frame_ms, audio_vad,
                           len(face_boxes), active_speaker)

        cv2.imshow("Active Speaker Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    if vad:
        vad.stop()
    if transcriber:
        transcriber.stop()
    if perf_logger:
        perf_logger.save()


if __name__ == "__main__":
    main()