import sys
import collections
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "TalkNet-ASD"))
from talkNet import talkNet as TalkNetModel

FACE_SIZE = 112
SAMPLE_RATE = 16000
VISUAL_WINDOW = 11


def load_talknet(model_path="pretrain_TalkSet.model", device="cpu"):
    path = Path(model_path)
    if not path.exists():
        import urllib.request
        print(f"[TalkNet] Downloading model...")
        url = "https://huggingface.co/spaces/TaoRuijie/TalkNet-ASD/resolve/main/pretrain_TalkSet.model"
        urllib.request.urlretrieve(url, str(path))

    model = TalkNetModel()
    model.loadParameters(str(path))
    model = model.eval()
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    print(f"[TalkNet] Loaded — device={device}")
    return model


def preprocess_face_crop(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (FACE_SIZE, FACE_SIZE)).astype(np.float32)
    gray = (gray - gray.mean()) / (gray.std() + 1e-6)
    return gray[np.newaxis]


class TalkNetStreamer:
    def __init__(self, model, window=VISUAL_WINDOW, device="cpu", smoothing=8):
        self.model = model
        self.window = window
        self.device = device
        self.video_buffers = {}
        self.score_history = {}
        self.current_scores = {}
        self.audio_buffer = collections.deque(
            maxlen=window * int(SAMPLE_RATE * 0.04)
        )
        self.smoothing = smoothing

    def update_video(self, face_id, face_crop_bgr):
        if face_id not in self.video_buffers:
            self.video_buffers[face_id] = collections.deque(maxlen=self.window)
            self.score_history[face_id] = collections.deque(maxlen=self.smoothing)

        self.video_buffers[face_id].append(preprocess_face_crop(face_crop_bgr))

        if len(self.video_buffers[face_id]) < 3:
            return 0.0

        raw_score = self._run_inference(face_id)
        self.score_history[face_id].append(raw_score)
        smoothed = float(np.mean(self.score_history[face_id]))
        self.current_scores[face_id] = smoothed
        return smoothed

    def update_audio(self, audio_chunk):
        for sample in audio_chunk.flatten():
            self.audio_buffer.append(float(sample))

    def get_score(self, face_id):
        return self.current_scores.get(face_id, 0.0)

    def reset(self, face_id=None):
        if face_id is None:
            self.video_buffers.clear()
            self.score_history.clear()
            self.current_scores.clear()
        else:
            self.video_buffers.pop(face_id, None)
            self.score_history.pop(face_id, None)
            self.current_scores.pop(face_id, None)

    @torch.no_grad()
    def _run_inference(self, face_id):
        frames = list(self.video_buffers[face_id])
        T = len(frames)

        video_tensor = torch.FloatTensor(
            np.stack([f[0] for f in frames], axis=0)
        ).unsqueeze(0)

        if self.device == "cuda":
            video_tensor = video_tensor.cuda()

        audio_features = self._build_audio_features(T)

        try:
            output = self.model.forward_stream(audio_features, video_tensor)
            score = torch.sigmoid(output).mean().item()
        except Exception:
            try:
                visual_input = video_tensor.squeeze(0).unsqueeze(1)
                output = self.model.visualEncoder(visual_input)
                score = torch.sigmoid(output).mean().item()
            except Exception:
                score = 0.0

        return float(np.clip(score, 0.0, 1.0))

    def _build_audio_features(self, num_frames):
        target_length = num_frames * 4
        if len(self.audio_buffer) < 100:
            features = torch.zeros(1, target_length, 13)
            return features.cuda() if self.device == "cuda" else features

        audio_array = np.array(list(self.audio_buffer), dtype=np.float32)
        try:
            import python_speech_features
            mfcc = python_speech_features.mfcc(
                audio_array, samplerate=SAMPLE_RATE,
                numcep=13, winlen=0.025, winstep=0.010
            )
            if len(mfcc) >= target_length:
                mfcc = mfcc[:target_length]
            else:
                padding = np.tile(mfcc[-1:], (target_length - len(mfcc), 1))
                mfcc = np.concatenate([mfcc, padding])
            features = torch.FloatTensor(mfcc).unsqueeze(0)
        except Exception:
            features = torch.zeros(1, target_length, 13)

        return features.cuda() if self.device == "cuda" else features