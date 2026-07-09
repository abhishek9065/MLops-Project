# DigitalOcean Deployment Guide

This project is designed to run first on a budget-friendly CPU Droplet. You do not need a GPU for this version because local embeddings are lightweight and hosted LLM calls can run over API.

Official references:

- Droplet pricing: https://docs.digitalocean.com/products/droplets/details/pricing/
- Basic Droplet use cases: https://docs.digitalocean.com/products/droplets/concepts/choosing-a-plan/
- Create a Droplet: https://docs.digitalocean.com/products/droplets/how-to/create/
- Firewalls: https://docs.digitalocean.com/products/networking/firewalls/
- Docker 1-Click option: https://docs.digitalocean.com/products/marketplace/catalog/docker/
- Nginx and Let's Encrypt: https://www.digitalocean.com/community/tutorials/how-to-secure-nginx-with-let-s-encrypt-on-ubuntu-22-04

## Recommended Starting Droplet

Start with a shared CPU Basic Droplet, Ubuntu 24.04 LTS or Ubuntu 22.04 LTS, and at least 2 GB RAM. If you run Grafana, Prometheus, Qdrant, Postgres, and Streamlit together, 2 GB is the practical minimum and 4 GB is more comfortable.

Avoid GPUs for this project unless you later self-host embedding or LLM models.

## Cost Notes

DigitalOcean bills Droplets while they exist, even when powered off, because reserved compute capacity remains allocated. Destroy unused Droplets to stop billing. Snapshots, volumes, backups, reserved IPs, and outbound transfer over plan limits can also cost money.

Use your $100 credit carefully:

- Create one small Droplet.
- Destroy test Droplets when finished.
- Avoid GPU Droplets.
- Avoid large volumes and snapshots until needed.
- Keep backups off for short demos unless you need them.
- Use hosted LLM APIs with strict usage limits.

## Create the Droplet

1. Go to DigitalOcean and create a Droplet.
2. Choose Ubuntu 24.04 LTS or Ubuntu 22.04 LTS.
3. Choose a Basic shared CPU plan.
4. Add your SSH key.
5. Add a tag such as `ragops`.
6. Create the Droplet.

## Configure Firewall

Allow only:

- SSH: `22`
- HTTP: `80`
- HTTPS: `443`

Do not expose Postgres, Qdrant, Prometheus, Grafana, or MLflow publicly unless you add authentication and firewall restrictions.

On the Droplet:

```bash
sudo ufw allow OpenSSH
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw --force enable
sudo ufw status
```

## Install Docker and System Packages

Copy and run:

```bash
bash deploy/setup_droplet.sh
```

Then log out and back in so Docker group membership applies.

## Clone and Configure

```bash
git clone https://github.com/YOUR_USERNAME/enterprise-ragops-platform.git
cd enterprise-ragops-platform
cp .env.example .env
nano .env
```

Set provider keys only if you want hosted LLMs:

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=...
```

For a free local demo:

```env
LLM_PROVIDER=local
LLM_MODEL=local-extractive-rag-v1
```

## Run Docker Compose

```bash
docker compose config
docker compose up -d --build
docker compose ps
```

Local ports on the Droplet:

- API: `http://127.0.0.1:8000`
- Streamlit: `http://127.0.0.1:8501`
- Prometheus: `http://127.0.0.1:9090`
- Grafana: `http://127.0.0.1:3000`

For production, Compose binds service ports to `127.0.0.1`. Expose only Nginx on 80/443 and keep service ports private.

## Nginx Reverse Proxy

```bash
sudo cp deploy/nginx/ragops.conf /etc/nginx/sites-available/ragops.conf
sudo ln -s /etc/nginx/sites-available/ragops.conf /etc/nginx/sites-enabled/ragops.conf
sudo nginx -t
sudo systemctl reload nginx
```

Edit `server_name` first:

```bash
sudo nano /etc/nginx/sites-available/ragops.conf
```

Use:

```text
server_name yourdomain.com;
```

The Docker frontend uses the internal Compose URL:

```env
API_BASE_URL=http://api:8000
```

External users can still reach the API through Nginx at:

```text
https://yourdomain.com/api
```

## HTTPS with Certbot

Point your domain DNS `A` record to the Droplet IP first.

Then:

```bash
sudo certbot --nginx -d yourdomain.com
sudo certbot renew --dry-run
```

## Health Checks

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/metrics | head
docker compose logs -f api
```

## Stop or Destroy

Stop containers:

```bash
docker compose down
```

Stop billing for the Droplet by destroying it in the DigitalOcean control panel when you are done.
