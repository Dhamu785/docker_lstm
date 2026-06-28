FROM python:3.14-bookworm

# Install system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN wget -qO- https://astral.sh/uv/install.sh | sh

# Make uv available
ENV PATH="/root/.local/bin:${PATH}"

# Create virtual environment
RUN uv venv /opt/lstm --system-site-packages

# Make the virtual environment the default Python
ENV VIRTUAL_ENV=/opt/lstm
ENV UV_PROJECT_ENVIRONMENT=/opt/lstm
ENV PATH="/opt/lstm/bin:/root/.local/bin:${PATH}"

# Default working directory
WORKDIR /workdir

CMD ["/bin/bash"]