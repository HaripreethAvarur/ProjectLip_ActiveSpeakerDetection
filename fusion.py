import collections
from typing import Optional

import numpy as np


class AudioVisualFusion:
    """
    Confidence-weighted fusion of TalkNet visual scores and microphone VAD.

    When audio VAD is confident (near 0 or 1), audio gets more weight.
    When VAD is ambiguous (near 0.5), visual scores get more weight.
    """

    def __init__(self,
                 base_visual_weight=0.60,
                 audio_confidence_threshold=0.75,
                 audio_silence_threshold=0.25,
                 confidence_shift=0.20,
                 smoothing_window=12):
        self.base_visual_weight = base_visual_weight
        self.audio_confidence_threshold = audio_confidence_threshold
        self.audio_silence_threshold = audio_silence_threshold
        self.confidence_shift = confidence_shift
        self.smoothing_window = smoothing_window
        self._score_buffers = {}
        self._latest_fused = {}

    def fuse(self, face_id, visual_score, audio_vad):
        alpha = self._compute_alpha(audio_vad)
        raw_score = alpha * visual_score + (1.0 - alpha) * audio_vad

        if face_id not in self._score_buffers:
            self._score_buffers[face_id] = collections.deque(maxlen=self.smoothing_window)
        self._score_buffers[face_id].append(raw_score)

        fused = float(np.mean(self._score_buffers[face_id]))
        self._latest_fused[face_id] = fused
        return fused

    def get_active_speaker(self, face_ids, visual_scores, audio_vad, min_score=0.35):
        if not face_ids:
            return None

        fused_scores = {
            fid: self.fuse(fid, visual_scores.get(fid, 0.0), audio_vad)
            for fid in face_ids
        }

        best_face = max(fused_scores, key=fused_scores.__getitem__)
        if fused_scores[best_face] < min_score:
            return None
        return best_face

    def get_fused_score(self, face_id):
        return self._latest_fused.get(face_id, 0.0)

    def reset(self):
        self._score_buffers.clear()
        self._latest_fused.clear()

    def _compute_alpha(self, audio_vad):
        if audio_vad >= self.audio_confidence_threshold or \
           audio_vad <= self.audio_silence_threshold:
            return max(0.0, self.base_visual_weight - self.confidence_shift)
        return min(1.0, self.base_visual_weight + self.confidence_shift)