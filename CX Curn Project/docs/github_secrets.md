# Adding GitHub Secrets safely (for CI/CD)

Secrets are encrypted values GitHub injects into workflow runs. **Never** put
these in code, `.env` committed to git, or the workflow YAML directly.

## Where to add them

Repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**.

## Secrets this project uses

| Secret name       | What it is                                              | How to get it |
| ----------------- | ------------------------------------------------------- | ------------- |
| `GITHUB_TOKEN`    | Auto-provided by GitHub. Used to push to GHCR.          | Nothing to do — it exists automatically. |
| `DROPLET_HOST`    | Your droplet's public IP, e.g. `164.92.x.x`.            | DigitalOcean dashboard. |
| `DROPLET_USER`    | SSH user, e.g. `deploy` (recommended) or `root`.        | You create it (see deploy guide). |
| `DROPLET_SSH_KEY` | **Private** SSH key that can log into the droplet.      | `cat ~/.ssh/id_ed25519` (the private key, whole file). |
| `GHCR_USER`       | Your GitHub username.                                   | — |
| `GHCR_PAT`        | Personal Access Token with `read:packages` so the droplet can `docker pull`. | GitHub → Settings → Developer settings → Tokens. |

## Generating a deploy SSH key (do this, don't reuse your personal key)

```bash
ssh-keygen -t ed25519 -C "github-deploy" -f ~/.ssh/churn_deploy
# Public key  -> add to the droplet's ~/.ssh/authorized_keys
# Private key -> paste into the DROPLET_SSH_KEY secret
```

## Golden rules

- A secret is **write-only** in the UI: once saved you can't read it back, only
  overwrite. Keep your own copy in a password manager.
- Secrets are **masked** in logs, but don't `echo` them — masking isn't perfect.
- Use a **dedicated, least-privilege** token/key (a deploy key, a `read:packages`
  PAT), not your all-powerful personal credentials.
- **Rotate** them if a laptop is lost or someone leaves the project.
- PRs from forks do **not** receive secrets — that's a security feature, not a bug.
