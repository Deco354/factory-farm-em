# fragile-compassion

Scoring infrastructure for the question: does animal-directed compassion in
language models degrade faster than human-safety behaviour under
emergent-misalignment (EM) fine-tuning?

This repo scores already-released EM LoRA adapters (the `ModelOrganismsForEM`
organisms on Hugging Face) on four instruments, all through
[Inspect](https://inspect.aisi.org.uk):

| task | measures | source |
|---|---|---|
| `fc_betley` | general misalignment (the dose) | Betley et al. free-form questions, judged with their verbatim prompts |
| `fc_anima` | animal-welfare moral reasoning | `inspect_evals/anima`, restricted to the 26 original English prompts, plus a refusal pass |
| `fc_strong_reject` | human-safety comparator, per-item continuous | `inspect_evals/strong_reject` |
| `fc_do_not_answer` | human-safety comparator, categorical | external `inspect-evals-do-not-answer` package |

There is no training code here; `sft_document_generation/` only generates candidate
SFT documents (see `SFT_DOCGEN_README.md`). See `CLAUDE.md` for the invariants and the plan
file for the research behind every design decision.

## Setup

### Macbook / Local Machine
```bash
uv python pin 3.12
uv sync --group dev                 # Mac: tests, planning, export
cp .env.example .env                # fill in OPENAI_API_KEY
```

### GPU Model serving box
```bash
uv python pin 3.12
uv sync --group dev --extra vllm    # Linux GPU box: also serves models
cp .env.example .env                # fill in OPENAI_API_KEY
```

On a rented Vast.ai box, `scripts/vast/up.sh` does all of this for you; see
[Rented GPU box (Vast.ai)](#rented-gpu-box-vastai).

## Run

The pipeline has three stages: **plan** what will run, **run** it, then **export**
the logs into per-item rows. Every stage reads the same three config files, so
the commands below only differ in what they do with them:

- `configs/models.yaml` — which base model and LoRA adapters to score, each pinned
  to a Hugging Face commit hash. The un-adapted base model is added automatically
  as the baseline; you never list it.
- `configs/judge.yaml` — the LLM that grades every response (currently OpenAI's
  `gpt-5.4-mini-2026-03-17` with reasoning off; see `docs/judge-selection.md`). It
  is set here and nowhere else; no scorer has a default judge.
- `configs/eval.yaml` or `configs/eval.smoke.yaml` — how much to run: epochs
  (repeat samples per question), generation temperature and token limits, and the
  Betley exclusion thresholds. The smoke profile has the same schema shrunk to
  finish in minutes; swap in `eval.yaml` for the real thing.

`--run-id` is a name you choose; logs for that run land in `logs/<run-id>/`.

### 1. Check the code (no GPU, no network, no API key)

```bash
uv run pytest
```

Unit tests for the pure parts only: the exclusion and misalignment rule, judge
reply parsing, the Betley YAML loader on a hand-written fixture, the refusal
asymmetry, and the export row builder. Nothing here calls a model.

### 2. Dry run: see exactly what would execute

```bash
uv run fc plan --models configs/models.yaml --judge configs/judge.yaml --eval configs/eval.smoke.yaml --run-id smoke-001
```

Prints every Inspect `eval_set` call without running anything: the model strings
(base plus each LORA adapter with its commit), the arguments that will be passed to
the vLLM server (`revision`, `enable_lora`, `max_lora_rank`), the log directory,
and each of the four tasks with its full argument list. Use it to confirm the
judge, epochs, and hashes before spending GPU time. Fully offline.

### 3. Run the evaluations

```bash
uv run fc run --models configs/models.yaml --judge configs/judge.yaml --eval configs/eval.smoke.yaml --run-id smoke-001
```

Executes the plan. For each base model it makes one `eval_set` call covering the
base and all its adapters across the four tasks. Inspect starts a vLLM server,
loads each adapter, generates responses, sends each response to the judge, and
writes one `.eval` log per (task, model) pair under `logs/smoke-001/<base>@<revision>/`.

- Needs a GPU for the models (see *Hardware topology*) and `OPENAI_API_KEY` in
  `.env` for the judge.
- Resumable: rerunning the identical command after a crash retries only the
  (task, model) pairs that did not finish.
- Every run's metadata records the config file hashes, the git commit, and the
  model and adapter commit hashes.

### 4. Export per-item rows

```bash
uv run fc export logs/smoke-001 --out results/smoke-001.jsonl --csv
```

Reads every `.eval` log under the directory and writes one row per
(model, benchmark, question, epoch). Each row carries:

- identity: adapter, base, both commit hashes, rank, domain, variant, benchmark;
- per-response token counts, plus the judge's token usage;
- every score key, flattened as `score__<scorer>__<key>` (e.g.
  `score__betley_judge__alignment`, `score__anima_scorer__overall`);
- `excluded` and `exclusion_reason`, after applying the rules: Betley responses
  judged CODE, REFUSAL, or incoherent are excluded; ANIMA refusals are excluded
  (a refusal is neither compassion nor its absence); human-safety refusals are
  kept, because there refusal is the correct behaviour.

JSONL is always written; `--csv` adds a CSV next to it. Add `--with-text` to
include the prompt and response text in each row. Scores are never averaged
here; analysis happens downstream.

### Other useful commands

```bash
uv run fc list-models --models configs/models.yaml   # expanded model list, including the auto-added base
uv run inspect view --log-dir logs/smoke-001         # Inspect's log viewer: per-sample transcripts and judge replies
```

### Running one task directly through Inspect

For debugging a single benchmark on a single model without the config files.
This is the same machinery `fc run` uses, spelled out by hand:

```bash
uv run inspect eval fragile_compassion/fc_betley \
  --model "vllm/unsloth/Qwen2.5-14B-Instruct:ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice@25ed05c042afdee9412e9132560cd49f0377ffad" \
  -M revision=facfb1bad6443964128be460ff6c98928a4ad4ab -M enable_lora=true -M max_lora_rank=32 -M max_model_len=4096 \
  -T source=first_plot -T judge=openai/gpt-5.4-mini-2026-03-17 -T judge_reasoning_effort=none -T epochs=2 \
  --log-dir logs/smoke-cli
```

- `fragile_compassion/fc_betley` — the task. The other three are `fc_anima`,
  `fc_strong_reject`, `fc_do_not_answer`.
- `--model vllm/<base>:<adapter>@<commit>` — Inspect's vLLM provider syntax: base
  model, then the LoRA adapter, then the adapter's commit hash. Drop the
  `:<adapter>@<commit>` part to run the unmodified base.
- `-M ...` — arguments forwarded to `vllm serve`: the base model's commit,
  LoRA enabled with ranks up to 32 (vLLM's default is 16, too small for the
  rank-32 adapters), and a 4096-token context.
- `-T ...` — arguments to the task itself: which Betley question file
  (`first_plot` or `preregistered`), the judge model (required, no default) and
  its settings (`judge_reasoning_effort`; `judge_temperature`, where `null` sends
  none), and epochs, i.e. how many responses to sample per question.
- `--log-dir` — where the `.eval` log is written. `fc export` works on it.

To check the plumbing with no GPU and no API key, pass `mockllm/model` as both
`--model` and `-T judge=`. Scores come out as NaN because the mock judge never
returns a number, but the dataset fetch, scorer, metrics, and log format are all
exercised end to end.

## Hardware topology

Inspect downloads each LoRA adapter locally and sends that local path to vLLM.
So run Inspect on the GPU box and let it start vLLM itself (recommended).

Unless told otherwise, Inspect starts `vllm serve` on `0.0.0.0` (every network
interface) with the well-known API key `inspectai`. On any machine you share, or
rent, set `VLLM_DEFAULT_SERVER_ARGS={"host": "127.0.0.1"}` and a random
`VLLM_API_KEY` in `.env` (see `.env.example`); `scripts/vast/box-setup.sh` writes both.

If vLLM must be remote, pre-register adapters by name:

```bash
hf download ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice \
  --revision 25ed05c042afdee9412e9132560cd49f0377ffad --local-dir adapters/medical-r32
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True vllm serve unsloth/Qwen2.5-14B-Instruct \
  --revision facfb1bad6443964128be460ff6c98928a4ad4ab --enable-lora --max-lora-rank 32 \
  --lora-modules medical-r32=adapters/medical-r32 --max-model-len 4096 --port 8000 --api-key inspectai
```

then set `VLLM_BASE_URL=http://<gpu-box>:8000/v1` and use the model string
`vllm/unsloth/Qwen2.5-14B-Instruct:medical-r32`. That server listens on every
interface, so replace `inspectai` with a random key (and set the same `VLLM_API_KEY`
where Inspect runs) unless the box is on a private network.

Qwen2.5-14B in bf16 needs ~30 GB for weights. One 80 GB GPU is comfortable; one
48 GB GPU works with `max_model_len=4096`. Do not quantise: it changes the model
under study.

## Rented GPU box (Vast.ai)

[Vast.ai](https://vast.ai) rents GPUs by the hour. Each rental, an *instance*, is a
Docker container on a machine owned by a third-party *host*. Three scripts in
`scripts/vast/` run a whole session from your laptop (macOS or Linux):

1. `up.sh` rents a box and sets it up.
2. You run `fc` on the box over SSH.
3. `down.sh` copies the results back and destroys the box.

Treat everything on a rented box as readable by the host's operator. Only two
credentials go onto the box, both made just for rented boxes:

- a judge API key that belongs to its own project, has a hard spend limit, and is easy
  to revoke;
- optionally, a read-only Hugging Face token. `up.sh` refuses a token that can do more
  than read.

Never put credentials in Vast environment variables or templates, since hosts can
read those too. Destroy the box when you are done.

### What it costs

- **Billing.** Storage bills from the moment a box is created; GPU time bills once it
  is running. Destroying the box stops both.
- **Why a fresh box each session.** A *stopped* box still bills for its disk and does
  not keep its GPU: someone else can rent it, and you cannot restart it until they
  finish.
- **No auto-shutdown.** A forgotten box bills until you destroy it or your credit runs
  out. When the credit runs out, Vast stops the box, then deletes it some time later.
- **Downloads bill separately.** The hourly price leaves them out, and hosts charged
  anywhere from $0 to $40 per TB on 2026-10-01. Each new box downloads about 45 GB,
  so setup costs $0 to $1.80 in downloads alone. `up.sh` ranks offers on both
  costs.
- **The hourly price includes the disk.** `up.sh` prices each offer with the disk it
  will rent (`FC_VAST_DISK`, 120 GB). Storage was about $0.05/hr of a 120 GB box on
  2026-10-01, and varies by host. The web console's prices match `up.sh`'s only when
  its disk filter is set to the same size.
- **Measured on the first smoke run** (2026-09-28, one 80 GB A100; prices vary by
  host):
  - ~45 GB of downloads, about $0.20;
  - ~15 minutes of setup GPU time, about $0.50;
  - about $2 an hour of GPU.

### One-time setup

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
    study will need several (`docs/judge-selection.md`, the note under Table 3).
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
anonymously: that works, but Hugging Face rate-limits it, and gated datasets (such as
the one TAC uses) can't be downloaded at all.

- On <https://huggingface.co/settings/tokens>, create a new token of type
  **Fine-grained**, named for example `vast-boxes`.
- Under Repositories, tick only **Read access to contents of all public gated repos
  you can access**, and leave every other box unticked. Public repos, such as the
  models, are readable by any token.
- For each gated dataset you need, accept its terms on the dataset's page, signed in to
  the same account. For TAC that's
  <https://huggingface.co/datasets/CompassioninMachineLearning/tac>.

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

**8. VS Code**, only if you use Remote-SSH. Add to your user settings. To open them,
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

**9. Check everything without renting anything:**

```bash
scripts/vast/up.sh --dry-run
```

This checks:

- the tools and your SSH config;
- that your commit is on GitHub;
- the judge key, with one real call that sees only `judge.env` (a fraction of a cent);
- the Hugging Face token, if you made one, is valid and read-only.

It then searches offers and prints what it would rent.

### Each session

On your laptop, from the repo:

```bash
scripts/vast/up.sh      # asks before renting; about 10-15 minutes until the box is ready
ssh vast-em             # or VS Code: Remote-SSH, host vast-em
```

The box checks out your current commit, so push it first:

- `up.sh` refuses a commit that is not on GitHub.
- It warns about uncommitted changes, since the box won't have them.
- If `up.sh` fails, or you stop it after renting, run it again. It resumes the same box
  instead of renting another.

On the box, always work inside tmux, so a dropped connection doesn't stop the run:

```bash
tmux new -s fc                 # after reconnecting: tmux attach -t fc
cd fragile-compassion
git log --oneline -1           # the commit you meant to run
uv run fc plan --eval configs/eval.smoke.yaml --run-id <run-id>
uv run fc run  --eval configs/eval.smoke.yaml --run-id <run-id>
```

`fc run` defaults to `configs/eval.yaml`, the full study, so always pass `--eval`.
To leave tmux without stopping the run, press Ctrl-b, then d.

When the run has finished, go back to your laptop:

```bash
scripts/vast/down.sh           # copies logs/ and results/ to logs/vast-<date>-<instance>/, then destroys the box
uv run fc export logs/vast-<date>-<instance>/logs/<run-id> --out results/<run-id>.jsonl --csv
```

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
| `FC_JUDGE_ENV` | `~/.config/fragile-compassion/judge.env` | the judge key file copied to the box |
| `FC_HF_ENV` | `~/.config/fragile-compassion/hf.env` | the optional read-only Hugging Face token file (step 7); no file means anonymous downloads |
| `FC_VAST_SSH_KEY` | `~/.ssh/vastai` | the private key for the box |
| `FC_VAST_QUERY` | 1 GPU > 70 GB, CUDA ≥ 13.0, Ampere or Hopper (A100/H100 class), x86, verified host with reliability > 0.98, direct SSH port, > 1 Gbps down | the `vastai search offers` filter. Blackwell workstation cards are excluded (untested with this vLLM), and so is Vast's datacenter-only tier, which has no A100s. Setting it replaces the whole query |
| `FC_VAST_QUERY_EXTRA` | none | clauses appended to the query. For the same field and operator the last clause wins, and `field=any` drops a filter, so `reliability>0.99` tightens a default, `disk_bw>1000` adds one, and `inet_down=any` removes one, without restating the rest |
| `FC_VAST_MAX_DPH` | `3.00` | the most it will ever rent at, in $/hr for the GPU plus storage for `FC_VAST_DISK`, even with `--yes`. Applies on top of `FC_VAST_QUERY` |
| `FC_VAST_HOURS` | `1` | GPU hours used to rank offers: the cheapest is the lowest hourly price × hours + 45 GB of downloads. Raise it for long runs, where the hourly price matters more |
| `FC_VAST_DISK` | `120` | disk in GB (about 60 GB used: 35 GB of models, ~10 GB venv, the image) |
| `FC_VAST_IMAGE` | `vastai/base-image:cuda-13.0.3-cudnn-devel-ubuntu24.04-2026-09-07` | the Docker image |

`up.sh --yes` answers every question for you, for scripted use. `down.sh --yes` skips its
countdown and answers its two questions.

### What is on the box

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

### Troubleshooting

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
| `hf download`: `Invalid filename '/root/.../activate'` | VS Code typed `source .venv/bin/activate` into your command; see step 8. |
| `up.sh`: not putting the token … on a rented box | The Hugging Face token in `hf.env` is invalid, or can do more than read (the line above names the permissions). Make a read-only one (step 7). Nothing was rented. |
| Setup log: `Rate limited. Waiting … before retry` | Anonymous Hugging Face downloads are rate-limited. They usually recover by themselves; a token (step 7) avoids it. |
| A gated dataset fails to load on the box (401 or "gated") | The box has no Hugging Face token, or the token's account hasn't accepted the dataset's terms (step 7). `box-setup.sh --check` says whether a token is present. |
| Inspect sits at a few percent, ~75 GB of GPU memory, 0% utilisation | vLLM is ready but the judge is failing, and Inspect retries a failing model call indefinitely. Check the judge key, and look for HTTP retries in Inspect's footer. |
| `fc run` shows nothing for a long time after starting vLLM | Inspect 0.3.263 sets no vLLM start timeout, so a stuck start waits forever. Check `nvidia-smi` and `pgrep -af vllm` in another tmux window. |
| `down.sh`: Vast still lists the instance | Destroy it in the console: <https://cloud.vast.ai/instances/>. |
