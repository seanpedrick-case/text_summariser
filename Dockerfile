# First stage: build dependencies
FROM public.ecr.aws/docker/library/python:3.11.9-slim-bookworm

# Install Lambda web adapter
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.3 /lambda-adapter /opt/extensions/lambda-adapter

# Install wget, git, curl
RUN apt-get update && \
    apt-get install -y wget git curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /src

COPY requirements.txt .

# Optimized dependency installation
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gradio==4.36.0

# Create a directory for the models and switch to user
RUN mkdir /model && \
    useradd -m -u 1000 user && \
    chown -R user:user /model
USER user

WORKDIR /home/user

# Download the GGUF model to local model/phi directory:
ENV REPO_ID "QuantFactory/Phi-3-mini-128k-instruct-GGUF"
ENV MODEL_FILE "Phi-3-mini-128k-instruct.Q4_K_M.gguf"

RUN python -c "from huggingface_hub import hf_hub_download; \
                hf_hub_download(repo_id='$REPO_ID', filename='$MODEL_FILE', local_dir='/model/phi')"

# Download the transformers-based models
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y git-lfs && \
    git lfs install

RUN git clone https://huggingface.co/stacked-summaries/flan-t5-large-stacked-samsum-1024 /model/stacked_t5 && \
    rm -rf /model/stacked_t5/.git && \
    git clone https://huggingface.co/pszemraj/long-t5-tglobal-base-16384-book-summary /model/long_t5 && \
    rm -rf /model/long_t5/.git


ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=$HOME/app \
    PYTHONUNBUFFERED=1 \
    GRADIO_ALLOW_FLAGGING=never \
    GRADIO_NUM_PORTS=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_THEME=huggingface \
    SYSTEM=spaces

# Switch back to root to copy the app files
USER root
WORKDIR /home/user/app

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/user/app

# Switch back to the user to run the app
USER user
CMD ["python", "app.py"]