# AdaptiveCoach — Streamlit app image, deployed to Hugging Face Spaces by
# the `deploy` job in .github/workflows/ci.yml. Also runs anywhere Docker does:
#   docker build -t adaptive-coach . && docker run -p 7860:7860 --env-file .env adaptive-coach
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/home/user \
    FASTEMBED_CACHE_PATH=/home/user/.cache/fastembed

# Hugging Face Spaces run the container as uid 1000.
RUN useradd -m -u 1000 user
WORKDIR /home/user/app
RUN chown user:user /home/user/app

# Dependencies first, so code changes don't reinstall them.
COPY --chown=user requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=user . .
USER user

# Bake the FAISS index (and the embedding model it needs) into the image, so
# the app starts without a multi-minute ingest. The index is built with the
# default local embeddings; keep EMBEDDING_PROVIDER unset (or "local") at
# runtime or queries won't match it. BUILD_INDEX=false skips this step for
# smoke-test builds that have no network access to the model.
ARG BUILD_INDEX=true
RUN if [ "$BUILD_INDEX" = "true" ]; then python -m rag.ingest; fi

EXPOSE 7860
CMD ["bash", "scripts/start_space.sh"]
