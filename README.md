# Lung & Colon Histopathology Classification

This project implements an adaptive hybrid fusion framework for five-class histopathology classification on the LC25000 dataset.

## Overview

- EfficientNet-B0 CNN backbone for deep image embeddings
- Handcrafted descriptors: colour moments, GLCM, LBP, Gabor filters, and HOG
- Learnable gated fusion for combining CNN and handcrafted feature streams
- Training, evaluation, and model checkpointing
- FastAPI inference service and Grad-CAM-style explanation endpoint
- Docker dev/prod environments and CI/CD automation

## Dataset layout

Place the dataset as follows:

```text
data/
└── lc25000/
    ├── colon_aca/
    ├── colon_n/
    ├── lung_aca/
    ├── lung_n/
    └── lung_scc/
```

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -m app.train --data-dir data/lc25000 --output-dir artifacts --epochs 10 --batch-size 16
```

## Run the API

```bash
export MODEL_PATH=artifacts/best.pt
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

Prediction example:

```bash
curl -F 'file=@sample.png' http://localhost:8000/predict
```

## Development environment (Docker)

```bash
docker compose --profile dev up --build
```

The dev API is exposed on http://localhost:8000

## Production environment (Docker)

```bash
cp .env.example .env
# update MODEL_PATH if necessary
MODEL_PATH=artifacts/best.pt docker compose --profile prod up --build -d
```

## Training and evaluation

```bash
python -m app.train --data-dir data/lc25000 --output-dir artifacts --epochs 10 --batch-size 16
python -m app.evaluate --data-dir data/lc25000 --checkpoint artifacts/best.pt --output-dir artifacts/evaluation
```

## Notes

- This is research software and not a medical device.
- Use a licensed dataset and keep patient/source separation in mind when splitting data.
- Output should be reported with the exact split and class-level metrics.
