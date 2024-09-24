# Use the official PyTorch Docker image as a base (includes CUDA and PyTorch)
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# Install required dependencies (add any additional system dependencies you need)
RUN apt update && apt install -y ffmpeg

# Create a non-root user with a home directory
RUN useradd -m -u 1000 user

# Switch to the new non-root user
USER user

# Set environment variables for the new user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set a working directory
WORKDIR $HOME/app

# Set the TRANSFORMERS_CACHE directory to be within the user's home directory
ENV TRANSFORMERS_CACHE=$HOME/cache

# Copy the app code and set ownership to the non-root user
COPY --chown=user . $HOME/app

# Install Python dependencies in the virtual environment
RUN python -m venv /home/user/venv
ENV PATH="/home/user/venv/bin:$PATH"

# Install pip dependencies within the virtual environment
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Run the training script
CMD ["python", "app.py"]
