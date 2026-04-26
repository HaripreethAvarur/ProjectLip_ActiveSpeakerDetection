import collections
import queue
import threading
import time
from typing import Optional

import numpy as np

SAMPLE_RATE = 16000


class VoiceActivityDetector:
    FRAME_SAMPLES = int(SAMPLE_RATE * 0.030)
    SMOOTHING_WINDOW = 15

    def __init__(self, mic_device=None, aggressiveness=2):
        self.mic_device = mic_device
        self._speech_prob = 0.0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._prob_history = collections.deque(maxlen=self.SMOOTHING_WINDOW)
        self._audio_callbacks = []

        try:
            import webrtcvad
            self._webrtc_vad = webrtcvad.Vad(aggressiveness)
            self._use_webrtc = True
            print("[VAD] Using WebRTC VAD")
        except Exception:
            self._webrtc_vad = None
            self._use_webrtc = False
            print("[VAD] Using RMS energy fallback")

        self._thread = threading.Thread(target=self._recording_loop, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def get_speech_probability(self):
        with self._lock:
            return self._speech_prob

    def add_audio_callback(self, callback_fn):
        self._audio_callbacks.append(callback_fn)

    def _recording_loop(self):
        try:
            import sounddevice as sd
        except ImportError:
            print("[VAD] sounddevice not installed")
            return

        frame_buffer = np.zeros(self.FRAME_SAMPLES, dtype=np.float32)
        buffer_index = 0

        def audio_callback(indata, frames, time_info, status):
            nonlocal frame_buffer, buffer_index
            for sample in indata[:, 0]:
                frame_buffer[buffer_index] = sample
                buffer_index += 1
                if buffer_index == self.FRAME_SAMPLES:
                    chunk = frame_buffer.copy()
                    self._process_frame(chunk)
                    for cb in self._audio_callbacks:
                        cb(chunk)
                    buffer_index = 0

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                blocksize=self.FRAME_SAMPLES // 4,
                device=self.mic_device, callback=audio_callback
            ):
                while not self._stop_event.is_set():
                    time.sleep(0.05)
        except Exception as e:
            print(f"[VAD] Stream error: {e}")

    def _process_frame(self, frame):
        if self._use_webrtc:
            try:
                pcm_bytes = (frame * 32767).astype(np.int16).tobytes()
                is_speech = float(self._webrtc_vad.is_speech(pcm_bytes, SAMPLE_RATE))
            except Exception:
                is_speech = 0.0
        else:
            rms = float(np.sqrt(np.mean(frame ** 2)))
            is_speech = min(1.0, rms / 0.04)

        self._prob_history.append(is_speech)
        with self._lock:
            self._speech_prob = float(np.mean(self._prob_history))


class WhisperTranscriber:
    def __init__(self, model_size="tiny", chunk_seconds=3.0,
                 mic_device=None, language="en"):
        print(f"[Whisper] Loading '{model_size}' model...")
        import whisper
        self._model = whisper.load_model(model_size)
        self._latest_transcript = ""
        self._lock = threading.Lock()
        self._audio_queue = queue.Queue(maxsize=3)
        self._stop_event = threading.Event()
        self.chunk_seconds = chunk_seconds
        self.mic_device = mic_device
        self.language = language
        self._record_thread = threading.Thread(target=self._record_loop, daemon=True)
        self._transcribe_thread = threading.Thread(target=self._transcribe_loop, daemon=True)
        print("[Whisper] Ready.")

    def start(self):
        self._record_thread.start()
        self._transcribe_thread.start()

    def stop(self):
        self._stop_event.set()

    def get_latest(self):
        with self._lock:
            return self._latest_transcript

    def _record_loop(self):
        import sounddevice as sd
        num_samples = int(SAMPLE_RATE * self.chunk_seconds)
        while not self._stop_event.is_set():
            try:
                audio = sd.rec(
                    num_samples, samplerate=SAMPLE_RATE,
                    channels=1, dtype="float32",
                    device=self.mic_device, blocking=True
                )
                if not self._audio_queue.full():
                    self._audio_queue.put(audio.flatten())
            except Exception as e:
                print(f"[Whisper] Recording error: {e}")
                time.sleep(0.5)

    def _transcribe_loop(self):
        while not self._stop_event.is_set():
            try:
                audio_chunk = self._audio_queue.get(timeout=1.0)
                result = self._model.transcribe(
                    audio_chunk, language=self.language,
                    fp16=False, verbose=False
                )
                transcript = result.get("text", "").strip()
                if transcript:
                    with self._lock:
                        self._latest_transcript = transcript
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Whisper] Transcription error: {e}")