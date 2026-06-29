FROM pytorch/pytorch:2.12.0-cuda12.6-cudnn9-devel

# Create virtual environment
RUN uv venv /opt/lstm --system-site-packages

# Make the virtual environment the default Python
ENV VIRTUAL_ENV=/opt/lstm
ENV UV_PROJECT_ENVIRONMENT=/opt/lstm
ENV PATH="/opt/lstm/bin:/root/.local/bin:${PATH}"
