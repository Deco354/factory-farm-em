# Rented GPU box (Vast.ai)

[Vast.ai](https://vast.ai) rents GPUs by the hour. Each rental, an *instance*, is a
Docker container on a machine owned by a third-party *host*. Three scripts in
`scripts/vast/` run a whole session from your laptop (macOS or Linux):

1. `up.sh` rents a box and sets it up.
2. You run `fc` on the box over SSH.
3. `down.sh` copies the results back and destroys the box.

The commands for a session are in the [README](../README.md), steps 2 to 6. This page
has the [credentials](#credentials) a session uses, the
[one-time setup](#one-time-setup) you do before the first session, and the details
behind each script.

Treat everything on a rented box as readable by the host's operator. Only credentials
made just for rented boxes go onto it:

- a judge API key that belongs to its own project, has a hard spend limit, and is easy
  to revoke;
- optionally, a read-only Hugging Face token. `up.sh` refuses a token that can do more
  than read;
- on a training box only, a token that can write to one private Hugging Face repo, for
  the adapters.

Never put credentials in Vast environment variables or templates, since hosts can
read those too. Destroy the box when you are done.

## What it costs

- **Billing.** Storage bills from the moment a box is created; GPU time bills once it
  is running. Destroying the box stops both.
- **Why a fresh box each session.** A *stopped* box still bills for its disk and does
  not keep its GPU: someone else can rent it, and you cannot restart it until they
  finish.
- **No auto-shutdown.** A forgotten box bills until you destroy it or your credit runs
  out. When the credit runs out, Vast stops the box, then deletes it some time later.
- **Downloads bill separately.** The hourly price leaves them out, and hosts charged
  anywhere from $0 to $40 per TB on 2026-10-01. A new box downloads the run's models
  plus about 10 GB of wheels: about 45 GB for the 14B set, so $0 to $1.80 in downloads
  alone, and about 85 GB for the 32B set. `up.sh` sums the pinned repos' sizes from
  Hugging Face and ranks offers on both costs.
- **The hourly price includes the disk.** `up.sh` sizes the disk from the same sum (the
  models plus 50 GB, 85 GB for the 14B set; `FC_VAST_DISK` overrides) and prices each
  offer with it. Storage was about $0.05/hr of a 120 GB box on 2026-10-01, and varies by
  host. The web console's prices match `up.sh`'s only when its disk filter is set to the
  same size.
- **Measured on the first smoke run** (2026-09-28, one 80 GB A100; prices vary by
  host):
  - ~45 GB of downloads, about $0.20;
  - ~15 minutes of setup GPU time, about $0.50;
  - about $2 an hour of GPU.

## Credentials

Every file a session reads a key or token from. The [one-time setup](#one-time-setup)
creates them, and the step numbers below are its steps. Only three put a secret on a
box: `judge.env`, `hf.env` and, on a training box, `hf-write.env`.

| file | holds | read by | must have | made in |
|---|---|---|---|---|
| `~/.config/vastai/vast_api_key` | the restricted Vast API key | `vastai`, so `up.sh` and `down.sh`; never goes on a box | only `instance_read`, `instance_write` and `misc`; mode 600 | step 3 |
| `~/.ssh/vastai` and `vastai.pub` | the SSH key for rented boxes | `ssh vast-em`, `up.sh`, `down.sh`; the public half goes on the Vast account and each box | a passphrase; used for nothing else, and the only key on the Vast account | step 4 |
| `~/.config/fragile-compassion/judge.env` | the judge key: `OPENAI_API_KEY`, or the variable for whichever provider `configs/judge.yaml` names | `up.sh`, which makes one real judge call with it before renting, then writes it into the box's `.env` for every judged task | its own project, a hard spend limit, an expiry, only `/v1/responses`; mode 600 (`up.sh` sets it) | step 6 |
| `~/.config/fragile-compassion/hf.env` | `HF_TOKEN`, read-only (optional) | `up.sh`, which checks it before renting, then writes it to the box's Hugging Face token file; downloads and TAC's and ANIMA's gated datasets on the box | fine-grained, with only **Read access to contents of all public gated repos you can access**; its account has accepted TAC's and ANIMA's terms; mode 600 (`up.sh` sets it) | step 7 |
| `~/.config/fragile-compassion/hf-write.env` | `HF_TOKEN`, the adapter-upload token (training only) | `up.sh --study` only, which checks it before renting, then writes it only to the box's `/root/.config/fc/hf-write-token` for `fc train upload`. `up.sh` without `--study` doesn't read it, and removes any copy from the box | fine-grained; its only permission beyond reading is `repo.write` on the study's `adapter_repo`, a private model repo; mode 600 (`up.sh` sets it) | step 8 |
| `~/.config/fragile-compassion/vast.env` | `FC_NAME=value` settings, never secrets (optional) | `up.sh` | only the names in [Settings](#settings), which `up.sh` checks | [Settings](#settings) |
| `.env` in the repo | the keys `fc` uses where it runs outside a rented box: `OPENAI_API_KEY`, and `HF_TOKEN` for TAC, ANIMA and `fc train plan`'s contamination check | `fc` on your laptop or your own Linux box; never copied to a rented box | `HF_TOKEN`'s account has accepted TAC's and ANIMA's terms | `.env.example`; README, [your own Linux box](../README.md#your-own-linux-box) |

## One-time setup

You need `git`, `python3`, `ssh` and [uv](https://docs.astral.sh/uv/) on your laptop.

**1. Vast account and credit.** Sign up at <https://cloud.vast.ai>, then add credit
under Billing. With a card saved, Vast charges it automatically for any negative
balance. A small prepaid balance and no saved card is the closest thing Vast has to a
spending cap: at zero, Vast stops the box and later deletes it.

**2. The Vast CLI:**

```bash
curl -fsSL https://vast.ai/install.sh | bash   # or: pip install vastai
vastai --version                                # the scripts were written against 1.8.2
```

**3. A restricted Vast API key.** It can search offers and create, inspect and destroy
instances. It cannot touch billing or account settings.

- On <https://cloud.vast.ai/manage-keys/?tab=api-keys>, create a new API key with only
  `instance_read`, `instance_write` and `misc` ticked.
- Then run:

  ```bash
  vastai set api-key <the new key>
  chmod 600 ~/.config/vastai/vast_api_key
  vastai search offers 'num_gpus=1 gpu_ram>70' --limit 3   # should list a few offers
  ```

If the console only offers full-access keys, create the restricted key with the CLI
instead. This uses a full key once without saving it:

```bash
printf '%s' '{"api": {"misc": {}, "instance_read": {}, "instance_write": {}}}' > /tmp/fc-perms.json
printf 'Full-access key: '; read -rs VAST_API_KEY; echo
VAST_API_KEY=$VAST_API_KEY vastai create api-key --name fc-em --permission_file /tmp/fc-perms.json
unset VAST_API_KEY; rm /tmp/fc-perms.json
```

**4. A dedicated SSH key for Vast.** Don't reuse your GitHub key:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/vastai -C vastai   # set a passphrase
cat ~/.ssh/vastai.pub                              # paste into the console: Keys → SSH Keys
```

Vast copies the account's keys into each box when it is created. `up.sh` also attaches
this key to every box it rents. Keep only this key on the account.

**5. SSH config.** Put these lines at the **very top** of `~/.ssh/config`, above every
`Host` block:

```
# Written by scripts/vast/up.sh for each box; ssh skips it until the first one exists.
Include ~/.ssh/vast-em.conf
```

`up.sh` rewrites that file for each box, so `ssh vast-em` always reaches the current
one. The entry it writes:

- uses only `~/.ssh/vastai` (`IdentitiesOnly yes`);
- never forwards your SSH agent;
- keeps Vast host keys in their own file, `~/.ssh/known_hosts_vast`, because Vast
  reuses IP:port pairs across boxes.

If you wrote a `Host vast-em` block by hand before, delete it: an earlier block wins
over the included one, and `up.sh` stops when `ssh vast-em` does not point at the new
box.

**6. A judge key only for rented boxes.** The box calls the judge in
`configs/judge.yaml` (currently OpenAI's `gpt-5.4-mini-2026-03-17`) with a key that
sits on the box for the whole session. Make it one you can afford to lose. On the
OpenAI platform:

- Create a separate project, for example `fc-rented-boxes`.
- Open <https://platform.openai.com/settings/> with the new project selected, then
  choose Limits:
  - allow only the model `gpt-5.4-mini-2026-03-17`;
  - under Spend, choose Edit spend limit, set $100 a month, and turn on **Enforce a
    hard limit**. Without it the limit only sends an email and the key keeps
    spending. With it, judge calls are refused (`429 project_spend_limit_exceeded`)
    for the rest of the month once the limit is reached, so raise it before a run
    that would cross it. Spend can overshoot slightly, since enforcement is not
    instantaneous ([OpenAI: spend limits](https://developers.openai.com/api/docs/guides/spend-limits)).
  - One full pass of `configs/eval.yaml` needs about $40 of judge calls, and the
    study will need several ([judge-selection.md](judge-selection.md), the note
    under Table 3).
- Under the project's API keys, create a key with an expiry date and **Restricted**
  permissions. Change only **Model capabilities**, and in this order:
  1. Set Model capabilities itself to Request.
  2. Expand it and set every endpoint except `/v1/responses` back to None. The row
     then shows "Mixed".

  Granting only `/v1/responses` fails with `Missing scopes: model.request`, because
  setting the parent row is what grants that permission. Leave every other row,
  List models included, at None: Inspect calls only the Responses API for this judge.

Keep the key on your laptop only:

```bash
mkdir -p ~/.config/fragile-compassion
printf 'OpenAI key for rented boxes: '; read -rs k; echo
printf 'OPENAI_API_KEY=%s\n' "$k" > ~/.config/fragile-compassion/judge.env; unset k
chmod 600 ~/.config/fragile-compassion/judge.env
```

The file holds `NAME=value` lines for whichever judge `configs/judge.yaml` names. If
the judge moves to another provider, put that provider's key variable here instead.
Revoke and replace the key whenever a box may have been compromised. After you change
`judge.env`, rerun `scripts/vast/up.sh` to put the new key on a box that is already
running; it replaces the old one.

**7. A Hugging Face token for rented boxes (optional).** Without one, the box downloads
anonymously: that works, but Hugging Face rate-limits it, and gated datasets (TAC's and
ANIMA's) can't be downloaded at all.

- On <https://huggingface.co/settings/tokens>, create a new token of type
  **Fine-grained**, named for example `vast-boxes`.
- Under Repositories, tick only **Read access to contents of all public gated repos
  you can access**, and leave every other box unticked. Public repos, such as the
  models, are readable by any token.
- For each gated dataset you need, accept its terms on the dataset's page, signed in to
  the same account. Approval is automatic. The evals need two:
  - TAC's, <https://huggingface.co/datasets/CompassioninMachineLearning/tac>;
  - ANIMA's, <https://huggingface.co/datasets/CompassioninMachineLearning/anima>.

Keep the token on your laptop only, in its own file:

```bash
printf 'Hugging Face token for rented boxes: '; read -rs k; echo
printf 'HF_TOKEN=%s\n' "$k" > ~/.config/fragile-compassion/hf.env; unset k
chmod 600 ~/.config/fragile-compassion/hf.env
```

`up.sh` reads the token only from this file, never from your shell or the project's
`.env`. It checks the token with one free call before renting, and refuses one that can
do more than read, such as write access or paid inference. It then writes the token to
Hugging Face's token file on the box (`~/.cache/huggingface/token`), so setup, vLLM,
Inspect and dataset loading all use it. Revoke and replace the token whenever a box may
have been compromised; rerunning `up.sh` puts the new one on a running box.

**8. An upload token for training boxes (only if you train).** `up.sh --study` puts
this token on a training box, so `fc train upload` can push adapters to one private
repo. An evaluation box never gets it. Make the repo first, then a token that can write
to it and nothing else:

- On <https://huggingface.co/new>, create a model repo and set it to **Private**. The
  study's `adapter_repo` (in `configs/training/studies/`) names it as `<owner>/<name>`.
- On <https://huggingface.co/settings/tokens>, create a new token of type
  **Fine-grained**, named for example `vast-adapter-upload`.
- Under Repositories permissions, select that repo and tick **Write access to
  contents/settings of selected repos**; read access to it is fine too. Leave **Interact
  with discussions / Open pull requests on selected repos**, and every other box,
  unticked: `up.sh` refuses a token that can do anything else, and names each extra
  permission it found.

Keep the token on your laptop only, in its own file:

```bash
printf 'Hugging Face upload token for training boxes: '; read -rs k; echo
printf 'HF_TOKEN=%s\n' "$k" > ~/.config/fragile-compassion/hf-write.env; unset k
chmod 600 ~/.config/fragile-compassion/hf-write.env
```

Check it without renting, once a study is planned
([training.md](training.md#on-a-rented-box-stage-2)):

```bash
scripts/vast/up.sh --study runs/<study> --dry-run
```

It should print `writes only <owner>/<name>: OK` and `<owner>/<name> exists and is
private: OK`. Revoke and replace the token whenever a training box may have been
compromised.

**9. VS Code**, only if you use Remote-SSH. Add to your user settings. To open them,
press Cmd+Shift+P (Ctrl+Shift+P on Linux) and run **Preferences: Open User Settings
(JSON)**. Keep any settings already there.

```json
"remote.SSH.enableAgentForwarding": false,
"git.terminalAuthentication": false,
"remote.autoForwardPorts": false,
"python.terminal.activateEnvironment": false
```

- The first two stop the box from using your laptop's SSH keys or GitHub login while
  you are connected.
- The third stops VS Code forwarding the box's ports to your laptop.
- The last stops the Python extension typing `source .venv/bin/activate` into the
  middle of a command you paste.

**10. Check everything without renting anything:**

```bash
scripts/vast/up.sh --dry-run
```

This checks:

- the tools and your SSH config;
- that your commit is on GitHub;
- the run's config files: `--eval` (default `configs/eval.yaml`), and the models and
  judge files the profile names, or `--models` and `--judge` if given;
- the judge key, with one real call to that judge that sees only `judge.env` (a
  fraction of a cent);
- the Hugging Face token, if you made one, is valid and read-only;
- that Hugging Face has every pinned model, and how big each one is.

It then prints what the box would be set up for: evaluation only, or training too
(`--study runs/<study>`, [training.md](training.md#on-a-rented-box-stage-2)). It lists the
config files it copies over, each model it pre-downloads, the disk it sized from them,
and where adapters upload. Then it searches offers and prints what it would rent. Once it
passes, carry on from the [README](../README.md), step 2.

## Session details

### Starting a session

`scripts/vast/up.sh` asks before renting, and takes about 10-15 minutes until the box is
ready. The box checks out your current commit, so push it first:

- `up.sh` refuses a commit that is not on GitHub.
- It warns about uncommitted changes, since the box won't have them. The exception is the
  run's three config files, which it copies to the box as they are on your laptop.
- If `up.sh` fails, or you stop it after renting, run it again. It resumes the same box
  instead of renting another.
- To change what a running box runs, edit the config files and run
  `scripts/vast/up.sh --eval <profile>` again. It resumes the box, copies the files over
  and downloads any models it doesn't have yet.

Connect with `ssh vast-em`, or VS Code's Remote-SSH with host `vast-em`. On the box,
always work inside tmux, so a dropped connection doesn't stop the run:

- `tmux new -s fc` to start; `tmux attach -t fc` after reconnecting.
- Ctrl-b, then d, leaves tmux without stopping the run.
- `git log --oneline -1` in `fragile-compassion/` shows the commit the box checked out.

### Ending a session

`scripts/vast/down.sh` copies the box's `logs/` and `results/` to
`logs/vast-<date>-<instance>/` on your laptop, then destroys the box.

`down.sh` doesn't ask before destroying. A box left running by mistake costs far more
than one destroyed by mistake, and its files are copied first anyway. Instead:

- **It warns first.** The moment it starts, it prints in red that it will destroy the
  box, then counts down 5 seconds. Press Ctrl-C at any point before "destroying
  instance" to stop it; the box keeps running, and it says so.
- **It still asks in two cases:** when something is still running on the box, since
  destroying it would kill that run, and when the box can't be reached, so nothing
  could be copied.
- **To copy without destroying:** run `scripts/vast/down.sh --keep`.

It then waits until Vast confirms the box is gone, and lists anything still on the
account.

Copy anything you made outside `logs/` and `results/` first, for example console output
you saved in `/root`. Destroying the box deletes it.

### Settings

`up.sh` has the settings below. Set one for a single run on the command line
(`FC_VAST_MAX_DPH=4 scripts/vast/up.sh`). To keep your own defaults without editing
the repo, put them in `~/.config/fragile-compassion/vast.env`:

```bash
# one FC_NAME=value per line; nothing is expanded; unknown names are an error
FC_VAST_MAX_DPH=2.50
FC_VAST_QUERY_EXTRA="reliability>0.99"
```

A variable set in your environment wins over the file, and `up.sh` prints which
settings it took from where. `FC_VAST_SETTINGS=<file>` points it at a different file.

| variable | default | what it sets |
|---|---|---|
| `FC_REF` | `HEAD` | the commit or branch the box checks out; must be on GitHub |
| `FC_REPO_URL` | `origin`, as https | the repo the box clones; must be public |
| `FC_JUDGE_ENV` | `~/.config/fragile-compassion/judge.env` | the judge key file ([credentials](#credentials)) |
| `FC_HF_ENV` | `~/.config/fragile-compassion/hf.env` | the read-only Hugging Face token file ([credentials](#credentials)); no file means anonymous downloads |
| `FC_VAST_SSH_KEY` | `~/.ssh/vastai` | the private key for the box |
| `FC_VAST_QUERY` | 1 GPU > 70 GB, CUDA ≥ 13.0, Ampere or Hopper (A100/H100 class), x86, verified host with reliability > 0.98, direct SSH port, > 1 Gbps down | the `vastai search offers` filter. Blackwell workstation cards are excluded (untested with this vLLM), and so is Vast's datacenter-only tier, which has no A100s. Setting it replaces the whole query |
| `FC_VAST_QUERY_EXTRA` | none | clauses appended to the query. For the same field and operator the last clause wins, and `field=any` drops a filter, so `reliability>0.99` tightens a default, `disk_bw>1000` adds one, and `inet_down=any` removes one, without restating the rest |
| `FC_VAST_MAX_DPH` | `3.00` | the most it will ever rent at, in $/hr for the GPU plus storage for `FC_VAST_DISK`, even with `--yes`. Applies on top of `FC_VAST_QUERY` |
| `FC_VAST_HOURS` | `1` | GPU hours used to rank offers: the cheapest is the lowest hourly price × hours + the box's downloads (its models plus ~10 GB of wheels). Raise it for long runs, where the hourly price matters more |
| `FC_VAST_DISK` | the models plus 50 GB (70 GB with `--study`), at least 60 | disk in GB. The default is an estimate from one box (106 GB used for ~108 GB of models and the venv); check `df -h /` on a box before relying on it |
| `FC_VAST_IMAGE` | `vastai/base-image:cuda-13.0.3-cudnn-devel-ubuntu24.04-2026-09-07` | the Docker image |
| `FC_BOX_MODELS` | none | more models to pre-download, on top of the run's: models.yaml paths and `repo@<commit>` pins, separated by spaces |
| `FC_HF_WRITE_ENV` | `~/.config/fragile-compassion/hf-write.env` | the upload token file for a `--study` box ([credentials](#credentials)); not read without `--study` |

`up.sh --study runs/<study>` sets the box up for a study planned with `fc train plan`: the
training envs, its base model, its upload token, and the study itself
([training.md](training.md#on-a-rented-box-stage-2)). It replaces the settings
`FC_BOX_TRAINING` and `FC_ADAPTER_REPO`, which `up.sh` now refuses. `up.sh --push
runs/<study>` copies a re-planned study to the recorded box. `up.sh --yes` answers every
question for you, for scripted use. `down.sh --yes` skips its countdown and answers its
two questions.

## What is on the box

`up.sh` rents the box in SSH-only mode with no startup script. In that mode Vast skips
the image's own entrypoint, so none of its web services start (Instance Portal,
Jupyter, Syncthing, TensorBoard), and SSH is the only public port.

`up.sh` then writes your Hugging Face token, if you have one, to the box's
`~/.cache/huggingface/token`, and copies `scripts/vast/box-setup.sh` from your checkout
onto the box and runs it. The script:

- installs tmux if it is missing, and a pinned uv;
- clones the repo at your commit;
- runs `uv sync --extra vllm`;
- downloads every pinned model into the Hugging Face cache (~35 GB), so `fc run`
  doesn't wait on downloads.

Its log is `/root/fc-setup.log`. Afterwards, `up.sh`:

- writes the judge key into the box's `.env`, replacing any earlier one;
- checks from a fresh login shell that every model loads from the cache offline;
- lists which sockets listen publicly. Expect only sshd, on port 22.

`box-setup.sh` also gives the box's `.env` a random per-box `VLLM_API_KEY`, plus:

```
VLLM_DEFAULT_SERVER_ARGS={"host": "127.0.0.1"}
VLLM_HOST_IP=127.0.0.1
GLOO_SOCKET_IFNAME=lo
```

By default, Inspect starts `vllm serve` on `0.0.0.0` with the well-known key
`inspectai`.

- The first line keeps the API server on localhost.
- The second sets the address vLLM advertises to its own processes.
- The third keeps vLLM's internal sockets on localhost. vLLM's process groups use
  PyTorch's Gloo backend, which otherwise listens on the container's Docker network
  address (172.17.x.x), where other renters' containers on the same host might reach
  it. `VLLM_HOST_IP` doesn't move these sockets: on 2026-10-06 they stayed on
  172.17.0.2 until this line was added.

Check all three while `fc run` is going, from a second tmux window (Ctrl-b, then c):

```bash
ss -ltnp | grep -i vllm        # every line should show 127.0.0.1
```

## Troubleshooting

| symptom | cause and fix |
|---|---|
| `up.sh`: commit … is not on GitHub yet | Push your branch; the box clones from GitHub. Or set `FC_REF` to a pushed commit. |
| `up.sh`: judge check failed | The key in `judge.env` is wrong, revoked or missing a permission (`Missing scopes`), or its project hit the spend limit (429 `…spend_limit_exceeded`). Nothing was rented. |
| `up.sh`: vastai is not logged in | `vastai set api-key <restricted key>` (step 3). |
| `up.sh`: no offers match | Nothing fits under `FC_VAST_MAX_DPH` right now. Retry later, raise the cap, or relax `FC_VAST_QUERY`. Filter with `>`, not `>=`: the web UI mishandled `>=` on `compute_cap`, which is capability × 100. |
| The web console shows other prices or offer numbers than `up.sh` | Set the console's disk filter to `FC_VAST_DISK` (120 GB): both sides price storage for their own disk size. An offer number is one free GPU slot on a machine, so the same machine can show a different number. Compare the `m:` (machine) and `host:` numbers instead; `up.sh` prints both. |
| The web console lists machines that `up.sh` doesn't | Vast's search API, which `up.sh` uses, bundles similar offers and returns one per bundle. The console is the complete list. So `up.sh` can miss a slightly cheaper machine: on 2026-10-01 it showed machine 67803 at $1.253/hr, while 149752, on the same host, was $1.217/hr. No `vastai` option turns bundling off; as of vastai 1.8.2, `search offers --new` and the API's `disable_bundling` both return 400. |
| `up.sh`: ssh vast-em resolves to … | The `Include` line is missing or below a `Host` block, or a hand-written `Host vast-em` block comes first (step 5). |
| `up.sh`: instance is exited, unknown or offline | The host failed. Run `down.sh`, then `up.sh` again for another offer. |
| `up.sh`: box-setup.sh failed | The end of its log says why, for example a host without internet access. Fix it and rerun `up.sh` (same box), or `down.sh` and rent another. |
| `Permission denied (publickey)` | The box doesn't have your key: it wasn't on the account when the box was created and attaching it failed, or the account holds a different key. |
| `Bad port '1.2.3.4:5678'` from every ssh command | IP and port on one line in `~/.ssh/config`. One bad line breaks all SSH, including GitHub. |
| Passphrase asked on every connection | The key is not in your SSH agent: `ssh-add ~/.ssh/vastai` (macOS: `ssh-add --apple-use-keychain ~/.ssh/vastai`). |
| `hf download`: `Invalid filename '/root/.../activate'` | VS Code typed `source .venv/bin/activate` into your command; see step 9. |
| `up.sh`: not putting the token … on a rented box | The Hugging Face token in `hf.env` is invalid, or can do more than read (the line above names the permissions). Make a read-only one (step 7). Nothing was rented. |
| `up.sh --study`: no upload token, not putting the token in `hf-write.env` …, or not setting up a box whose adapters could not be uploaded | The upload token is missing, can do more than write to the study's `adapter_repo` (the line above names what else), or that repo is missing or public. Make or fix it as in step 8. Nothing was rented. |
| Setup log: `Rate limited. Waiting … before retry` | Anonymous Hugging Face downloads are rate-limited. They usually recover by themselves; a token (step 7) avoids it. |
| A gated dataset fails to load on the box (401 or "gated") | The box has no Hugging Face token, or the token's account hasn't accepted the dataset's terms (step 7). `box-setup.sh --check` says whether a token is present. |
| Inspect sits at a few percent, ~75 GB of GPU memory, 0% utilisation | vLLM is ready but the judge is failing, and Inspect retries a failing model call indefinitely. Check the judge key, and look for HTTP retries in Inspect's footer. |
| `fc run` shows nothing for a long time after starting vLLM | Inspect 0.3.263 sets no vLLM start timeout, so a stuck start waits forever. Check `nvidia-smi` and `pgrep -af vllm` in another tmux window. |
| `down.sh`: Vast still lists the instance | Destroy it in the console: <https://cloud.vast.ai/instances/>. |
