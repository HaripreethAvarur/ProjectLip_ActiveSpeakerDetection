# Active Speaker Detection + Captions
### Masters Project — ML for Sensors, Wearables & IoT

---

## Folder structure

```
v2/
├── TalkNet-ASD/           ← cloned from GitHub
├── pretrain_TalkSet.model ← auto-downloaded by setup.sh
├── setup.sh
├── live_demo.py           ← RUN THIS for the demo
├── talknet_inference.py
├── audio_modules.py
├── fusion.py
├── quantize_benchmark.py
├── ablation_study.py
└── results/               ← auto-created
```

---

## Commands — in this exact order

### 1. Clone TalkNet (already done)
```bash
git clone https://github.com/TaoRuijie/TalkNet-ASD
```

### 2. Run setup
```bash
bash setup.sh
```
Installs all dependencies + downloads pretrained model (~200MB).

### 3. Verify on a test video
```bash
cp /any/video.mp4 TalkNet-ASD/demo/test.mp4
cd TalkNet-ASD
python demoTalkNet.py --videoName test
cd ..
# Output: TalkNet-ASD/demo/test/pyavi/video_out.avi
# Open it — green box on speaking face = working
```

### 4. Live demo
```bash
python live_demo.py
```
- GREEN box = active speaker
- ORANGE box = silent
- Score bars: vis / aud / fus
- Caption bar = Whisper transcript
- Press Q to quit

### 5. Quantization benchmark (for report)
```bash
python quantize_benchmark.py --rpi-mode
# → results/quantization_report.txt
```

### 6. Ablation study (for report)
```bash
python ablation_study.py --video TalkNet-ASD/demo/test.mp4
# → results/ablation_report.txt
```

### 7. RPi deployment
```bash
# Copy to RPi
scp -r v2/ pi@raspberrypi.local:~/

# On RPi
bash setup.sh
python live_demo.py --threads 4 --no-audio --log results/rpi_perf.csv
```

---

## Options for live_demo.py

```bash
--camera 1          different camera index
--mic 2             different microphone
--list-mics         show available mics
--no-audio          visual ASD only (faster)
--device cuda       use GPU
--whisper base      larger/better Whisper model
--min-score 0.25    more sensitive detection
--log results/p.csv save per-frame perf data
--threads 4         restrict CPU threads (RPi)
```

---

## What to write in your report

| Section | Source |
|---------|--------|
| Architecture diagram | TalkNet backbone + your fusion layer |
| Quantization table | results/quantization_report.txt |
| Ablation table | results/ablation_report.txt |
| RPi performance | results/rpi_perf.csv |

### Your contributions
1. Live streaming inference wrapper (TalkNet was offline-only)
2. Confidence-weighted audio-visual fusion
3. Whisper caption integration
4. INT8 quantization + RPi deployment
5. Ablation study

---

## Troubleshooting

**`No module named talkNet`**
→ Make sure `TalkNet-ASD/` exists in the same folder as `live_demo.py`

**Model download fails**
```bash
# Download manually with wget:
wget -O pretrain_TalkSet.model \
  "https://huggingface.co/spaces/TaoRuijie/TalkNet-ASD/resolve/main/pretrain_TalkSet.model"
```

**webrtcvad fails**
```bash
pip install webrtcvad-wheels
```

**No faces detected**
→ Better lighting, face within 1.5m, try `--min-score 0.2`

**Green box flickers**
→ Raise `--min-score 0.4`