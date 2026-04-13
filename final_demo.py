import argparse
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "TalkNet-ASD"))

FONT = cv2.FONT_HERSHEY_SIMPLEX
SPEAKER_COLORS = {0: (0, 210, 0), 1: (0, 140, 255), 2: (255, 60, 120), 3: (255, 200, 0)}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="demo_input.mp4")
    parser.add_argument("--talknet-dir", default="TalkNet-ASD/demo")
    parser.add_argument("--out", default="final_demo.mp4")
    parser.add_argument("--whisper", default="tiny")
    parser.add_argument("--no-whisper", action="store_true")
    return parser.parse_args()


def transcribe_video(video_path, model_size):
    print("[Whisper] Extracting audio...")
    audio_path = "/tmp/whisper_audio.wav"
    os.system(f'ffmpeg -i "{video_path}" -ar 16000 -ac 1 -y "{audio_path}" -loglevel quiet')

    import whisper
    print(f"[Whisper] Loading '{model_size}' model...")
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, fp16=False, verbose=False)
    segments = result.get("segments", [])
    print(f"[Whisper] {len(segments)} segments transcribed")

    try:
        os.remove(audio_path)
    except Exception:
        pass

    return segments


def load_talknet_results(pywork_dir):
    with open(f"{pywork_dir}/tracks.pckl", "rb") as f:
        tracks = pickle.load(f)
    with open(f"{pywork_dir}/scores.pckl", "rb") as f:
        scores = pickle.load(f)
    print(f"[Tracks] {len(tracks)} face tracks loaded")
    return tracks, scores


def build_frame_map(tracks, scores):
    frame_map = {}
    for track_id, (track, score_array) in enumerate(zip(tracks, scores)):
        frame_indices = track["track"]["frame"]
        bounding_boxes = track["track"]["bbox"]

        for i, (frame_idx, bbox) in enumerate(zip(frame_indices, bounding_boxes)):
            frame_idx = int(frame_idx)
            score = float(score_array[i]) if i < len(score_array) else -999.0
            if frame_idx not in frame_map:
                frame_map[frame_idx] = []
            frame_map[frame_idx].append({
                "track_id": track_id,
                "box": (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                "score": score,
                "speaking": score > 0,
            })

    print(f"[Tracks] Frame map covers {len(frame_map)} frames")
    return frame_map


def assign_speakers_to_segments(segments, frame_map, fps):
    print("\n[Speaker attribution]")
    enriched_segments = []
    for segment in segments:
        start_frame = int(segment["start"] * fps)
        end_frame   = int(segment["end"]   * fps)

        speaker_votes = {}
        for frame_idx in range(start_frame, end_frame + 1):
            for track in frame_map.get(frame_idx, []):
                if track["speaking"]:
                    tid = track["track_id"]
                    speaker_votes[tid] = speaker_votes.get(tid, 0) + 1

        speaker_id = max(speaker_votes, key=speaker_votes.__getitem__) if speaker_votes else None
        enriched_segments.append({**segment, "speaker_id": speaker_id})

        speaker_label = f"Speaker {speaker_id+1}" if speaker_id is not None else "unknown"
        print(f"  [{segment['start']:.1f}s-{segment['end']:.1f}s] "
              f"{speaker_label}: {segment['text'].strip()[:60]}")

    return enriched_segments


def get_caption_at_time(segments, current_time):
    for segment in segments:
        if segment["start"] <= current_time <= segment["end"]:
            return segment["text"].strip(), segment.get("speaker_id")
    for segment in reversed(segments):
        if current_time <= segment["end"] + 1.5:
            return segment["text"].strip(), segment.get("speaker_id")
    return "", None


def draw_speaker_box(frame, x1, y1, x2, y2, color, label, score, is_speaking):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = max(0,x1), max(0,y1), min(w-1,x2), min(h-1,y2)
    if x2 <= x1 or y2 <= y1:
        return

    if is_speaking:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        corner_len = max(10, (x2 - x1) // 6)
        for cx, cy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(frame, (cx, cy), (cx + dx*corner_len, cy), color, 4)
            cv2.line(frame, (cx, cy), (cx, cy + dy*corner_len), color, 4)
        tag = f"{label}  SPEAKING"
        (tw, th), _ = cv2.getTextSize(tag, FONT, 0.62, 1)
        cv2.rectangle(frame, (x1, y1-th-14), (x1+tw+10, y1), color, -1)
        cv2.putText(frame, tag, (x1+5, y1-6), FONT, 0.62, (0, 0, 0), 2, cv2.LINE_AA)
        bar_height = y2 - y1
        filled = int(bar_height * min(abs(score) / 5.0, 1.0))
        cv2.rectangle(frame, (x2+4, y1), (x2+12, y2), (50, 50, 50), -1)
        cv2.rectangle(frame, (x2+4, y2-filled), (x2+12, y2), color, -1)
    else:
        dim_color = tuple(int(c * 0.4) for c in color)
        cv2.rectangle(frame, (x1, y1), (x2, y2), dim_color, 1)
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.50, 1)
        cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+6, y1), dim_color, -1)
        cv2.putText(frame, label, (x1+3, y1-4), FONT, 0.50, (180, 180, 180), 1, cv2.LINE_AA)


def draw_caption_bar(frame, caption_text, speaker_id):
    if not caption_text:
        return
    h, w = frame.shape[:2]
    speaker_color = SPEAKER_COLORS.get(speaker_id % len(SPEAKER_COLORS), (180, 180, 180)) \
                    if speaker_id is not None else (180, 180, 180)
    speaker_prefix = f"Speaker {speaker_id+1}" if speaker_id is not None else "?"
    display_text = caption_text[-75:] if len(caption_text) > 75 else caption_text

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-54), (w, h), (8, 8, 8), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame, (0, h-54), (7, h), speaker_color, -1)
    cv2.putText(frame, f"{speaker_prefix}:", (14, h-28),
                FONT, 0.60, speaker_color, 1, cv2.LINE_AA)
    cv2.putText(frame, display_text, (14, h-8),
                FONT, 0.60, (255, 255, 255), 1, cv2.LINE_AA)


def draw_hud(frame, frame_idx, total_frames, fps):
    elapsed_seconds = frame_idx / max(fps, 1)
    minutes = int(elapsed_seconds // 60)
    seconds = int(elapsed_seconds % 60)
    cv2.putText(frame, f"  {minutes:02d}:{seconds:02d}  frame {frame_idx}/{total_frames}",
                (8, 24), FONT, 0.44, (140, 140, 140), 1, cv2.LINE_AA)


def main():
    args = parse_args()

    video_stem = Path(args.video).stem
    pywork_dir = f"{args.talknet_dir}/{video_stem}/pywork"
    talknet_output_video = f"{args.talknet_dir}/{video_stem}/pyavi/video_out.avi"

    if not Path(talknet_output_video).exists():
        print(f"[Error] TalkNet output not found: {talknet_output_video}")
        print(f"Run first:")
        print(f"  cd TalkNet-ASD")
        print(f"  python demoTalkNet.py --videoName {video_stem} --videoFolder demo")
        return

    segments = []
    if not args.no_whisper:
        segments = transcribe_video(args.video, args.whisper)

    tracks, scores = load_talknet_results(pywork_dir)
    frame_map = build_frame_map(tracks, scores)

    cap = cv2.VideoCapture(talknet_output_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if segments:
        segments = assign_speakers_to_segments(segments, frame_map, fps)

    print(f"\n[Render] {frame_width}x{frame_height} @ {fps:.1f}fps  {total_frames} frames")
    cap = cv2.VideoCapture(talknet_output_video)
    tmp_output = "/tmp/asd_no_audio.mp4"
    writer = cv2.VideoWriter(tmp_output, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (frame_width, frame_height))

    last_speaking_frame = {}
    persist_duration = int(fps * 1.0)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_idx / fps

        for track in frame_map.get(frame_idx, []):
            if track["speaking"]:
                last_speaking_frame[track["track_id"]] = frame_idx

        for track in frame_map.get(frame_idx, []):
            track_id = track["track_id"]
            frames_since_speaking = frame_idx - last_speaking_frame.get(track_id, -9999)
            is_speaking = track["speaking"] or (frames_since_speaking <= persist_duration)
            color = SPEAKER_COLORS.get(track_id % len(SPEAKER_COLORS), SPEAKER_COLORS[0])
            x1, y1, x2, y2 = track["box"]
            draw_speaker_box(frame, x1, y1, x2, y2, color,
                             f"Speaker {track_id+1}", track["score"], is_speaking)

        caption_text, caption_speaker = get_caption_at_time(segments, current_time)
        draw_caption_bar(frame, caption_text, caption_speaker)
        draw_hud(frame, frame_idx, total_frames, fps)

        writer.write(frame)

        if frame_idx % 150 == 0:
            pct = frame_idx / max(total_frames, 1) * 100
            print(f"  {frame_idx}/{total_frames} ({pct:.0f}%)")

        frame_idx += 1

    cap.release()
    writer.release()

    print("[Audio] Muxing original audio...")
    result = os.system(
        f'ffmpeg -i "{tmp_output}" -i "{args.video}" '
        f'-c:v copy -c:a aac -map 0:v:0 -map 1:a:0 '
        f'-shortest -y "{args.out}" -loglevel quiet'
    )

    if result == 0:
        print(f"[Done] {args.out}")
    else:
        import shutil
        shutil.copy(tmp_output, args.out)
        print(f"[Done] {args.out} (audio mux failed — video only)")

    try:
        os.remove(tmp_output)
    except Exception:
        pass


if __name__ == "__main__":
    main()