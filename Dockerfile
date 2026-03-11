FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the mock tools and dependencies
COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
# Copy the rest of the execution code
COPY tools/ ./tools/
COPY core/ ./core/
COPY data/ ./data/
COPY run_benchmark.py .

# Command to run the automated benchmark loop
CMD ["python", "run_benchmark.py"]