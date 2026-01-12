---
title: Luminark Deepfake Detection
emoji: 🛡️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: true
license: mit
---

# Luminark API

**AI-powered deepfake video detection** using an ensemble of 6 neural network models.

## 🚀 Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/infer` | Analyze video → verdict |
| `POST` | `/explain` | Analyze with detailed XAI |

## 📊 Models Used

- **VideoMAE** - Video transformer (Microsoft)
- **WavLM** - Audio embeddings (Microsoft)
- **Spatial** - Frame-level CNN (EfficientNet)
- **Temporal** - Motion consistency (CNN-LSTM)
- **Frequency** - DCT analysis
- **Physiological** - rPPG signals

## 🔐 Authentication

Include `X-API-Key` header with your API key.

## 📝 Example

```bash
curl -X POST https://isvohi-luminark.hf.space/infer \
  -H "X-API-Key: your_key" \
  -F "video=@test.mp4"
```

---

Built with ❤️ by [Vikas Sharma](https://github.com/IsVohi)
