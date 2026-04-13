# Project Lip - Active Speaker Detection

Detects who is speaking in a video using TalkNet + Whisper captions + audio-visual fusion.

**Output:** Video with green box on active speaker, speaker-attributed captions ("Speaker 1: ...").

---

## Setup

0. Setup Environment: 
```bash
conda create --name env_projectlip python=3.10
conda activare env_projectlip
```

1. Clone TalkNet:
```bash
git clone https://github.com/TaoRuijie/TalkNet-ASD
```

2. Fix numpy compatibility:
```bash
sed -i 's/np\.int\b/int/g' TalkNet-ASD/model/faceDetector/s3fd/box_utils.py
```

3. Install dependencies:
```bash
pip install torch torchvision opencv-python openai-whisper gdown scenedetect pandas yt-dlp
conda install -c conda-forge ffmpeg -y
```

---

## Get a Demo Video

4. Download and trim a YouTube video (two people talking works best):
```bash
yt-dlp -f "best[ext=mp4][height<=480]" "https://www.youtube.com/watch?v=lhFU5H5KPFE" -o "demo_full.mp4"
ffmpeg -i demo_full.mp4 -ss 00:00:10 -t 00:01:30 -c copy -y demo_input.mp4
rm demo_full.mp4
```
(I have used that video, because thats the best I found. I also used it because I am not able to use my camera for live demo)

---

## Run

5. Test TalkNet on their sample video:
```bash
cd TalkNet-ASD
python demoTalkNet.py --videoName 001 --videoFolder demo
cd ..
```

6. Run TalkNet on your video:
```bash
cp demo_input.mp4 TalkNet-ASD/demo/demo_input.mp4
cd TalkNet-ASD
python demoTalkNet.py --videoName demo_input --videoFolder demo
cd ..
```

7. Generate final demo video (boxes + captions + audio):
```bash
python final_demo.py --video demo_input.mp4
```
Output: `final_demo.mp4`

8. Run quantization benchmark:
```bash
python quantize_benchmark.py --rpi-mode
```
Output: `results/quantization_report.txt`

---

## Files

| File | Description |
|---|---|
| `talknet_inference.py` | Streaming inference wrapper around TalkNet |
| `audio_modules.py` | VAD + Whisper background threads |
| `fusion.py` | Confidence-weighted audio-visual fusion |
| `live_demo.py` | Live webcam demo (needs camera) |
| `final_demo.py` | Offline demo on a video file |
| `quantize_benchmark.py` | INT8 quantization + edge deployment benchmark |
