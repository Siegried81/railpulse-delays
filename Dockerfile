# railpulse-delays — batch delay study.
# Runs run_analysis.py, writes results/ and figures/, then exits.
# The dataset is NOT in this repo (it lives in ../railpulse-genai/data), so it
# is mounted at /data at run time and passed as the DATA_DIR argument.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /app

# Install dependencies first so editing source does not reinstall packages
# (pip layer stays cached as long as requirements.txt is unchanged).
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Mount the data and keep the outputs on the host:
#   docker run --rm \
#     -v "$PWD/../railpulse-genai/data:/data" \
#     -v "$PWD/results:/app/results" \
#     -v "$PWD/figures:/app/figures" railpulse-delays
# The /data CMD argument overrides run_analysis.py's default DATA_DIR.
ENTRYPOINT ["python", "run_analysis.py"]
CMD ["/data"]
