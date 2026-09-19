# Running the agent on a Nebius Compute VM

The agent is a lightweight orchestrator — all inference happens on Token
Factory — so a small **CPU-only** VM is enough. This is the same shape a
partner would use for a scheduled evaluation job or an internal service.

## 1. Create the VM

In the Nebius console → **Compute** → **Create virtual machine**:

- **Image**: Ubuntu 24.04 LTS
- **Size**: 2–4 vCPU, 8 GB RAM is plenty (no GPU needed)
- **Network**: public IP for SSH, outbound internet access (the VM calls
  `api.tokenfactory.nebius.com`)
- Add your SSH public key

## 2. Bootstrap

SSH in and run the bootstrap script:

```bash
export REPO_URL=https://github.com/<you>/nebius-migration-engineer.git
curl -fsSL "https://raw.githubusercontent.com/<you>/nebius-migration-engineer/main/scripts/bootstrap_vm.sh" | bash
```

It installs Python + git, clones the repo, creates a venv, installs the
package, and copies `.env.example` to `.env`.

## 3. Configure and verify

```bash
cd ~/nebius-migration-engineer
nano .env        # set NEBIUS_API_KEY
./.venv/bin/tf-migrate preflight
```

Never commit `.env`; on a shared VM prefer an instance-scoped secret store or
environment variables injected by your scheduler.

## 4. Run

```bash
./.venv/bin/tf-migrate run workloads/summarization.yaml
```

Reports and evidence land in `runs/<timestamp>-<workload>/`.

## 5. Optional: scheduled evaluations

To re-validate a migration nightly (model catalogs and prices move), add a
cron entry:

```
0 3 * * * cd $HOME/nebius-migration-engineer && ./.venv/bin/tf-migrate run workloads/summarization.yaml >> cron.log 2>&1
```

## Notes for production hardening

- Put the API key in a proper secret manager, not a dotfile.
- Pin dependency versions (`pip freeze > requirements.lock`) for reproducible
  deploys.
- If you expose this as a service, add auth in front — the agent spends real
  inference money per run.
