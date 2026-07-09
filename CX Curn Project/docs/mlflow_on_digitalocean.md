# Running MLflow on a DigitalOcean Droplet (reference)

> You'll execute this for real in **Phase 7**. It's documented here so the
> concept is clear now: *the same code that logged to SQLite locally will log to
> a remote server by changing one environment variable.*

## Why run a remote MLflow server?

Locally, `sqlite:///mlflow.db` is a file on your laptop. That's useless for a
team or for CI: a GitHub Actions run, your teammate, and the production API all
need to see the **same** experiments and the **same** Model Registry. A remote
MLflow **tracking server** is that shared source of truth.

Production-grade MLflow has three parts:

| Part                | Local (now)        | Remote (Phase 7)                     |
| ------------------- | ------------------ | ------------------------------------ |
| Backend store (DB)  | SQLite file        | **PostgreSQL** (metrics, params, registry) |
| Artifact store      | local `mlartifacts/` | DigitalOcean **Spaces** (S3) or a droplet volume |
| Tracking server     | none (direct file) | `mlflow server` behind Nginx         |

## Option A — Quick: MLflow server on the Droplet with docker-compose

`docker-compose.mlflow.yml` (we'll add this to the repo in Phase 5):

```yaml
services:
  mlflow-db:
    image: postgres:16
    environment:
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: ${MLFLOW_DB_PASSWORD}
      POSTGRES_DB: mlflow
    volumes:
      - mlflow_pgdata:/var/lib/postgresql/data

  mlflow-server:
    image: ghcr.io/mlflow/mlflow:v2.16.2
    depends_on: [mlflow-db]
    command: >
      mlflow server
      --host 0.0.0.0 --port 5000
      --backend-store-uri postgresql://mlflow:${MLFLOW_DB_PASSWORD}@mlflow-db:5432/mlflow
      --artifacts-destination /mlartifacts
      --serve-artifacts
    ports: ["5000:5000"]
    volumes:
      - mlflow_artifacts:/mlartifacts

volumes:
  mlflow_pgdata:
  mlflow_artifacts:
```

Bring it up on the droplet:

```bash
export MLFLOW_DB_PASSWORD='choose-a-strong-password'
docker compose -f docker-compose.mlflow.yml up -d
```

## Point your training at it

On your laptop / in CI, change **one** variable — no code changes:

```bash
export MLFLOW_TRACKING_URI=http://<droplet-ip>:5000
python -m src.training.train
```

## Security notes (do this, don't skip)

- **Never** expose port 5000 to the whole internet unauthenticated. Put it
  behind Nginx with HTTP Basic Auth (Phase 7) or restrict by firewall/VPN.
- Keep `MLFLOW_DB_PASSWORD` in DigitalOcean as an environment secret, never in git.
- Use a DigitalOcean firewall to allow 5000 only from your IP / the app droplet.

## Common mistakes

- **Registry errors with a file backend.** The Registry needs Postgres/MySQL/SQLite.
  A remote server started with `--backend-store-uri file:...` will refuse to register.
- **Artifacts "not found" from the client.** Use `--serve-artifacts` so clients
  fetch artifacts *through* the server, instead of needing direct S3/disk access.
- **Forgetting the port.** `http://ip` defaults to port 80; MLflow is on 5000
  unless you front it with Nginx.
