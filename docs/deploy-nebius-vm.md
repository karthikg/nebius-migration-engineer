# Running the agent on a Nebius Compute VM

The agent is a lightweight orchestrator — all inference happens on Token
Factory — so a small **CPU-only** VM is enough. This is the same shape a
partner would use for a scheduled evaluation job or an internal service.

## 1. Create the VM

### Option A: Nebius CLI (what we actually used)

Gotchas learned in the field:

- **Quota**: fresh tenants have `compute.instance.non-gpu.vcpu = 0` in some
  regions (e.g. eu-north1, us-central1) and 200 in others (uk-south1,
  eu-west1, …). Check before you pick a region:
  `nebius quotas quota-allowance list --parent-id <tenant-id>`
- **SSH key**: cloud-init is the only way in. A passphrase-protected key
  won't work for unattended automation — use a dedicated passphrase-less
  deploy key.

```bash
# cloud-init: create the login user with your public key(s)
cat > cloudinit.yaml <<EOF
#cloud-config
users:
  - name: ubuntu
    groups: sudo
    shell: /bin/bash
    sudo: "ALL=(ALL) NOPASSWD:ALL"
    ssh_authorized_keys:
      - $(cat ~/.ssh/<your-key>.pub)
EOF

nebius compute instance create \
  --parent-id <project-id> \
  --name migration-agent \
  --resources-platform cpu-d3 \
  --resources-preset 4vcpu-16gb \
  --boot-disk-attach-mode read_write \
  --boot-disk-managed-disk-name migration-agent-boot \
  --boot-disk-managed-disk-type network_ssd \
  --boot-disk-managed-disk-size-gibibytes 64 \
  --boot-disk-managed-disk-source-image-family-image-family ubuntu24.04-driverless \
  --network-interfaces '[{"name":"eth0","subnet_id":"<subnet-id>","ip_address":{},"public_ip_address":{}}]' \
  --cloud-init-user-data "$(cat cloudinit.yaml)"
```

Get the subnet with `nebius vpc subnet list --parent-id <project-id>`; the
public IP appears in `nebius compute instance get` once RUNNING. Expect
sshd to accept connections a few minutes *after* the state says RUNNING.

### Option B: Web console

In the Nebius console → **Compute** → **Create virtual machine**:

- **Image**: Ubuntu 24.04 LTS
- **Size**: 2–4 vCPU, 8 GB RAM is plenty (no GPU needed)
- **Network**: public IP for SSH, outbound internet access (the VM calls
  `api.tokenfactory.nebius.com`)
- Add your SSH public key

## 2. Bootstrap

SSH in and run the bootstrap script:

```bash
export REPO_URL=https://github.com/karthikg/nebius-migration-engineer.git
curl -fsSL "https://raw.githubusercontent.com/karthikg/nebius-migration-engineer/main/scripts/bootstrap_vm.sh" | bash
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
