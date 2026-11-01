# Inspection API. Needs models/deploy/ first:
#   uv run python -m defect_detection.export deploy --run models/resnet18-lr0.001
# Build and run:
#   docker build -t casting-inspection .
#   docker run --rm -p 8000:8000 casting-inspection

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/srv/src:/srv \
    MODEL_DIR=/srv/models/deploy

WORKDIR /srv

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# only what serving needs, the training code pulls in torch
COPY src/defect_detection/__init__.py src/defect_detection/inference.py src/defect_detection/
COPY app/__init__.py app/main.py app/
COPY models/deploy/ models/deploy/

RUN useradd --create-home inspector
USER inspector

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
