# Deploying to DigitalOcean (budget-friendly, ~$100 credit)

This is a realistic, cost-conscious deployment. **No GPU** — churn inference is
CPU-only and cheap. The whole stack (API + MLflow + Prometheus + Grafana) fits
comfortably on a small droplet.

## Cost planning

| Choice                        | Recommendation                         | Approx cost |
| ----------------------------- | -------------------------------------- | ----------- |
| Droplet size                  | **Basic, 2 vCPU / 2 GB RAM** (Regular) | ~$12–18/mo  |
| Region                        | Closest to you / your users            | —           |
| Backups                       | Optional (+20%)                        | skip early  |

With $100 credit, a $12–18/mo droplet lasts ~5–8 months. Start small; you can
resize later.

> Money-saver: **destroy the droplet when you're done demoing** and recreate it
> from these scripts + CI in minutes. You only pay while it exists.

---

## Step 1 — Create the droplet

1. DigitalOcean → **Create → Droplets**.
2. Image: **Ubuntu 24.04 LTS**.
3. Plan: **Basic → Regular → 2 GB / 2 vCPU**.
4. Authentication: **SSH key** (paste your public key — far safer than passwords).
5. Hostname: `churn-mlops`. Create.

Copy the **public IP** once it's ready.

## Step 2 — First login + provisioning

```bash
ssh root@<droplet-ip>
# paste the setup script (or scp it up), then:
bash setup_droplet.sh
```

This installs Docker + Compose, creates a `deploy` user, and enables the firewall.
Log back in as the non-root user:

```bash
ssh deploy@<droplet-ip>
```

## Step 3 — Get the code + config

```bash
git clone https://github.com/<you>/mlops-churn-prediction.git ~/mlops-churn-prediction
cd ~/mlops-churn-prediction
cp .env.prod.example .env
nano .env       # set API_IMAGE to your GHCR image, set GRAFANA_PASSWORD
```

## Step 4 — Log in to GHCR + run

```bash
echo "<GHCR_PAT>" | docker login ghcr.io -u <github-username> --password-stdin
docker compose -f docker-compose.prod.yml pull
docker compose -f docker-compose.prod.yml up -d
docker compose -f docker-compose.prod.yml ps
```

Check it:

```bash
curl http://localhost:8000/health
```

From your laptop, `http://<droplet-ip>:8000/docs` should now work (port 8000 is
open only if you allow it — see below; the recommended path is via Nginx on 80/443).

## Step 5 — Put Nginx in front (recommended)

```bash
sudo apt-get install -y nginx
sudo cp deploy/nginx/churn-api.conf /etc/nginx/sites-available/churn-api
sudo ln -s /etc/nginx/sites-available/churn-api /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
```

Now the API is reachable on `http://<droplet-ip>/` (port 80). Keep 8000 closed to
the public — only Nginx needs it.

## Step 6 — HTTPS with Certbot (needs a domain)

Point an `A` record for `your-domain.com` at the droplet IP, then:

```bash
sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

Certbot edits the Nginx config to add TLS and sets up **auto-renewal**. Your API
is now on `https://your-domain.com/`.

---

## After the first deploy: CI does it for you

Once `DROPLET_HOST`, `DROPLET_USER`, `DROPLET_SSH_KEY`, `GHCR_USER`, `GHCR_PAT`
are set as GitHub secrets (see `github_secrets.md`), every push to `master` will
test → build → push → SSH in → `docker compose pull && up -d`. Zero manual steps.

## Common mistakes

- **Exposing MLflow (5000) or Grafana (3000) publicly with no auth.** Restrict
  them: bind to localhost, use the firewall, or put them behind Nginx basic-auth.
  Only the API (via Nginx 80/443) should be public.
- **Running everything as root.** Use the `deploy` user.
- **Forgetting the firewall.** `ufw` should allow only SSH/80/443.
- **2 GB RAM too tight?** If the trainer OOMs, resize the droplet or reduce
  `n_estimators` / dataset size. Inference itself is light.
- **Leaving the droplet running.** Destroy it between demos to save credit.
