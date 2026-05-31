FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (build-essential is useful for C/C++ extensions if needed, e.g. for XGBoost and SHAP compilation support)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port 8888 for Jupyter notebooks
EXPOSE 8888

# Default command: Start Jupyter Notebook on 0.0.0.0 so it is accessible from the host
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
