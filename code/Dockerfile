# We use the official PyTorch CUDA 12.1 runtime as the base image for out-of-the-box GPU support
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Install basic system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv (The blazing fast Rust-based Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh

# Set working directory
WORKDIR /app

# Copy dependency manifests first to leverage Docker layer caching!
COPY pyproject.toml uv.lock* ./

# Install Python dependencies directly into the container's system environment
# (Since the container itself is an isolated environment, we don't need a virtualenv inside it)
RUN uv pip install --system -r pyproject.toml

# Copy the rest of the application code
COPY . .

# Set PYTHONPATH so absolute imports (e.g. from src.polymarket_gnn...) work natively
ENV PYTHONPATH=/app

# Default command
CMD ["/bin/bash"]
