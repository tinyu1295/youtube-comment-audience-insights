# Deployment

The prediction API is a stateless Flask service. It loads the model once per
worker at boot and holds no request state, so it scales by running more
instances behind a load balancer.

## Local

```bash
docker compose up --build
curl http://localhost:8000/
```

Or without Docker:

```bash
pip install -r requirements.txt
gunicorn --bind 127.0.0.1:8000 --workers 2 flask_api.main:app
```

The development server (`python -m flask_api.main`) is for local iteration only.
It is single-threaded and its debug mode exposes an interactive Python console,
so it must not face a network.

## AWS EC2

This is how the service was deployed.

### 1. Instance

A `t3.small` running Amazon Linux 2023 is sufficient — the model is ~4 MB and
inference is CPU-bound but light. Security group:

| Port | Source | Purpose |
|---|---|---|
| 22 | your IP only | SSH |
| 8000 | load balancer SG, or your IP | API |

Do not open 8000 to `0.0.0.0/0` unless the API is intentionally public; it has
no authentication (see [Hardening](#hardening)).

### 2. Install and run

```bash
sudo dnf install -y docker git
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user   # log out and back in

git clone https://github.com/tinyu1295/youtube-comment-audience-insights.git
cd youtube-comment-audience-insights
docker compose up -d --build
```

`restart: unless-stopped` in the compose file brings the container back after a
reboot or a crash.

### 3. Point the extension at it

Set the backend URL in the extension's settings prompt to the instance's public
DNS, and add the matching origin to `host_permissions` in `manifest.json`.

## MLflow tracking server

Training and registration log to whatever `MLFLOW_TRACKING_URI` names, which
defaults to a local `./mlruns` directory so a fresh clone works offline. To use
a remote server:

```bash
export MLFLOW_TRACKING_URI=http://your-tracking-host:5000
dvc repro
```

Run the tracking server itself with:

```bash
mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri postgresql://... \
  --default-artifact-root s3://your-bucket/
```

MLflow ships with no authentication. Bind it to a private subnet or put it
behind an authenticating proxy — an exposed tracking server lets anyone read and
overwrite the model registry.

## Hardening

Deliberately not implemented, and what production would need:

- **Authentication.** The API is unauthenticated. Add an API key or put it
  behind a gateway before exposing it.
- **Rate limiting.** Nothing stops a client from posting unbounded comment
  batches; `/generate_wordcloud` in particular is CPU-heavy.
- **TLS.** Terminate HTTPS at a load balancer. Chrome will block plaintext HTTP
  from a published extension.
- **Request size limits.** Set `MAX_CONTENT_LENGTH`; a large payload currently
  gets vectorised and densified in memory.
- **Monitoring.** No metrics or alerting are wired up. Prediction latency and
  class distribution drift are the two worth watching first.
