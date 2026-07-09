#!/usr/bin/env bash
# Run this ONCE on a fresh Ubuntu 24.04 droplet to prepare it for deployment.
#   ssh root@<droplet-ip>
#   bash setup_droplet.sh
#
# It installs Docker + Compose, creates a non-root deploy user, and enables a
# basic firewall. Idempotent-ish: safe to re-run.
set -euo pipefail

echo "==> Updating system packages"
apt-get update && apt-get upgrade -y

echo "==> Installing Docker Engine + Compose plugin"
apt-get install -y ca-certificates curl gnupg
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  > /etc/apt/sources.list.d/docker.list
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "==> Creating non-root deploy user"
if ! id deploy >/dev/null 2>&1; then
  adduser --disabled-password --gecos "" deploy
  usermod -aG docker deploy
  mkdir -p /home/deploy/.ssh
  # Copy root's authorized_keys so you can SSH in as `deploy` with the same key.
  cp /root/.ssh/authorized_keys /home/deploy/.ssh/ 2>/dev/null || true
  chown -R deploy:deploy /home/deploy/.ssh
  chmod 700 /home/deploy/.ssh
fi

echo "==> Configuring firewall (SSH, HTTP, HTTPS only)"
apt-get install -y ufw
ufw allow OpenSSH
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable

echo "==> Done. Log in as: ssh deploy@<droplet-ip>"
docker --version
docker compose version
