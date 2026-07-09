# GitHub Actions Secrets

The CI/CD workflow is in:

```text
.github/workflows/ci-cd.yml
```

CI always runs:

- dependency installation
- tests
- RAG evaluation gate
- retrieval quality gate
- Docker image build

Deployment runs only when:

- branch is `main`
- event is `push`
- repository variable `ENABLE_DROPLET_DEPLOY` is set to `true`

## Required Deployment Secrets

Add these in GitHub:

```text
Settings -> Secrets and variables -> Actions
```

Secrets:

- `DO_HOST`: Droplet public IP or hostname.
- `DO_USER`: SSH user, usually `root` or a sudo user.
- `SSH_PRIVATE_KEY`: private key matching a public key installed on the Droplet.
- `OPENAI_API_KEY`: optional, only if using OpenAI.
- `GEMINI_API_KEY`: optional, only if using Gemini.

Variable:

- `ENABLE_DROPLET_DEPLOY=true`

## Deployment Assumption

The workflow assumes the repo already exists on the Droplet at:

```text
~/enterprise-ragops-platform
```

Create it once:

```bash
git clone https://github.com/YOUR_USERNAME/enterprise-ragops-platform.git ~/enterprise-ragops-platform
cd ~/enterprise-ragops-platform
cp .env.example .env
```

Then future pushes to `main` can deploy through SSH.

