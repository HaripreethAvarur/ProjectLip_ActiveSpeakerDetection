# Active Speaker Detection — Recorded Video Pipeline (GPU + TalkNet)

Detects who is speaking in a pre-recorded video using TalkNet for visual active speaker detection, WebRTC VAD for audio, and OpenAI Whisper for transcription. Outputs an annotated MP4 with colored bounding boxes around the active speaker and speaker-attributed captions ("Speaker 1: ...").

---

## File Layout

```
ProjectLip_ActiveSpeakerDetection-main/
├── final_demo.py            # Main script: TalkNet + Whisper → annotated MP4
├── live_demo.py             # Live webcam pipeline (GPU, TalkNet)
├── talknet_inference.py     # TalkNet model loader and streaming inference
├── audio_modules.py         # VAD (WebRTC/RMS fallback) + Whisper threads
├── fusion.py                # Confidence-weighted audio-visual fusion
├── evaluate_asd.py          # Evaluation suite (full / simulation / benchmark)
├── quantize_benchmark.py    # INT8 quantization + Raspberry Pi edge benchmark
├── eval_results/            # Pre-computed evaluation results (JSON, CSV, TXT)
└── results/                 # Pre-computed quantization benchmark outputs
```

---

## Setup

```bash
# Create and activate environment
conda create --name env_projectlip python=3.10
conda activate env_projectlip

# Clone TalkNet
git clone https://github.com/TaoRuijie/TalkNet-ASD

# Fix numpy compatibility in TalkNet
sed -i 's/np\.int\b/int/g' TalkNet-ASD/model/faceDetector/s3fd/box_utils.py

# Install dependencies
pip install torch torchvision opencv-python openai-whisper gdown scenedetect pandas yt-dlp
conda install -c conda-forge ffmpeg -y
```

---

## 1. Recorded Video Demo

### Step 1 — Get the demo video

```bash
yt-dlp -f "best[ext=mp4][height<=480]" "https://www.youtube.com/watch?v=lhFU5H5KPFE" -o "demo_full.mp4"
ffmpeg -i demo_full.mp4 -ss 00:00:10 -t 00:01:30 -c copy -y demo_input.mp4
rm demo_full.mp4
```

### Step 2 — Run TalkNet face detection and scoring

```bash
cp demo_input.mp4 TalkNet-ASD/demo/demo_input.mp4
cd TalkNet-ASD
python demoTalkNet.py --videoName demo_input --videoFolder demo
cd ..
```

This produces face tracks and per-frame speaking scores saved under `TalkNet-ASD/demo/demo_input/pywork/`.

### Step 3 — Generate annotated output video

```bash
python final_demo.py --video demo_input.mp4
```

Output: `final_demo.mp4` — annotated video with bounding boxes and speaker-attributed captions.

Optional flags:
```bash
python final_demo.py --video demo_input.mp4 --whisper base --out my_output.mp4
python final_demo.py --video demo_input.mp4 --no-whisper   # skip transcription
```

---

## 2. Live Webcam Demo (GPU)

```bash
python live_demo.py --device cuda --camera 0 --log results/live_run.csv
```

Press **Q** to quit. The `--log` flag saves a per-frame CSV for evaluation.

Optional flags:
```bash
python live_demo.py --device cpu --camera 0          # CPU fallback
python live_demo.py --list-mics                      # list microphone devices
python live_demo.py --mic 1 --min-score 0.35         # specify mic, adjust threshold
python live_demo.py --no-audio                       # visual-only, no VAD/Whisper
```

---

## 3. Evaluation

Pre-computed results are already in `eval_results/`. To reproduce:

```bash
# Full evaluation on the recorded video (uses audio energy as pseudo ground truth)
python evaluate_asd.py --mode full --video demo_input.mp4 --out eval_results/
# Output: eval_results/eval_full.json, eval_full_summary.csv, eval_full_report.txt

# Simulation baseline (no video needed, synthetic data)
python evaluate_asd.py --mode simulation --speakers 2 --frames 500 --out eval_results/
# Output: eval_results/eval_simulation_report.txt

# Latency benchmark
python evaluate_asd.py --mode benchmark --frames 200 --out eval_results/
```

To evaluate a live run log:

```bash
python evaluate.py results/live_run.csv
# Output: results/live_run_eval.txt
```

---

## 4. Quantization Benchmark (Raspberry Pi 4 simulation)

```bash
python quantize_benchmark.py --rpi-mode
```

Pre-computed results are already in `results/`. This restricts PyTorch to 4 CPU threads to simulate RPi 4 compute, applies INT8 dynamic quantization to the TalkNet visual encoder, and reports latency and throughput for FP32 vs INT8.

Output: `results/quantization_report.txt`, `results/benchmark.csv`

---

## Key Results

| Metric | Value |
|---|---|
| Precision | 0.5935 |
| Recall | 0.9167 |
| F1 Score | 0.7205 |
| Accuracy | 0.5733 |
| Face Detection Rate | 1.00 |
| Mean FPS | 37.3 |
| Mean Latency | 26.8 ms |
| FP32 → INT8 speedup | ~1.0x (Conv3D bottleneck) |
| Model size (FP32 / INT8) | 52.9 MB / 52.9 MB |

---

## Repository

https://github.com/HaripreethAvarur/ProjectLip_ActiveSpeakerDetection
