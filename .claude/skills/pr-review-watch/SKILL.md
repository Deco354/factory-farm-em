---
name: pr-review-watch
description: Watch a pull request for incoming review feedback, then triage and act on it. Use whenever the user asks to poll/watch/check a PR for a review, wait for CI or a reviewer, "let me know when the review comes in", "respond to the review comments", "address the feedback on the PR", or handle change requests on a PR they own. Verifies each review claim against the code before acting, applies the ones that are right, and escalates disagreements to the user instead of silently complying or silently ignoring. Never posts comments, replies, or reviews on the PR itself — that is always the human's to do.
---

# Watching and answering PR review feedback

> Org copy: the canonical version of this skill lives in `sentfutures/devops`
> under `skills/pr-review-watch/`. It is copied into each repo's
> `.claude/skills/` at install; if you improve it, improve the devops copy too.

## Two rules that override everything else

**1. You never post on the PR.** No comments, no replies, no reviews, no resolving threads, no dismissing reviews. All of that is the human's. You may *draft* reply text for them to post — drafting is fine, posting is not.

The reason isn't noise, it's honesty: an agent commenting under the author's account makes a PR look like a human weighed the review when none did. Keeping the poster human keeps the record truthful about who actually decided. Attribution notes are a weaker fix, because they can be forgotten — a ban can't be half-followed.

**2. Never silently comply, and never silently ignore.** A review comment has exactly three honest outcomes, all of which end in a report to the user rather than a post:

1. **It's right** → fix it, push, and tell the user what changed and where.
2. **It's wrong, or you disagree** → **stop and bring it to the user.** Don't quietly skip it, and don't argue with the reviewer anywhere.
3. **It's a question, not a change request** → answer it *to the user*, who can relay it.

Reviewers are often right, and bot reviewers are wrong often enough that compliance-by-default is a real hazard: it produces confident code changes justified by nothing. Verify every claim against the actual code before you touch anything.

## Step 1: check whether the feedback is already there

Do this before arming any watcher. Reviews frequently land before the user asks you to watch for them, and arming a monitor for a past event means waiting forever.

```bash
gh pr view <N> --repo <owner/repo> \
  --json state,reviewDecision,mergeable,reviews,comments,statusCheckRollup
```

`reviewDecision` reports the branch-protection review state, not the reviews
themselves: on a repo whose protection requires reviews it reads `APPROVED` /
`CHANGES_REQUESTED` / `REVIEW_REQUIRED` (bot approvals included), but on a
repo without required reviews it stays EMPTY no matter what sits in
`reviews` — verified both ways on this org's repos. Empty is therefore not
"no feedback": read the `reviews` array itself. If feedback exists, skip to
Step 3.

## Step 2: waiting, if it genuinely hasn't arrived

Pick by how many notifications you need:

**One notification when the review lands** — Bash with `run_in_background` and a loop that exits on the condition. Pick the condition by what you are waiting for:

*The review bot* (the usual case) — wait on its check leaving `pending`, then read the reviews. This works on every repo, protected or not:

```bash
until bucket=$(gh pr checks <N> --repo <owner/repo> --json name,bucket \
  -q '.[] | select(.name | contains("claude-review")) | .bucket'); \
  [ -n "$bucket" ] && [ "$bucket" != "pending" ]; do
  sleep 60
done
```

*A human reviewer, on a repo whose branch protection requires reviews* — `reviewDecision` is reliable there, and only there:

```bash
until [ -n "$(gh pr view <N> --repo <owner/repo> --json reviewDecision -q .reviewDecision)" ]; do
  sleep 60
done
```

Never use that loop on a repo without required reviews: `reviewDecision` stays empty there no matter what gets posted — an APPROVED bot review included — and a comment-only review never sets it anywhere, so the loop hangs forever after the reviewer has actually spoken. Universal fallback for every other case — poll the reviews list itself for a new bodied review (add a `submitted_at` filter against when you started waiting if the PR already carries older reviews):

```bash
until [ -n "$(gh api repos/<owner>/<repo>/pulls/<N>/reviews --paginate \
  | jq -r -s 'add // [] | [.[] | select((.body|length) > 0)] | last | .state // ""')" ]; do
  sleep 60
done
```

**One notification per new comment, indefinitely** — the `Monitor` tool with `persistent: true`, polling `gh api repos/<owner>/<repo>/issues/<N>/comments?since=<ts>` and emitting a line per new comment. Only worth it for an active back-and-forth.

Cadence: 60s+ for a bot review (usually 1–5 min), 10–30 min for a human. Don't poll a remote API faster than 30s. If you're in a `/loop`, use `ScheduleWakeup` with a delay matched to the reviewer, not a tight poll.

Also watch CI: a red test check is feedback too, and usually more urgent than a comment.

## Step 3: gather every feedback surface

Review bodies and inline comments live in different places. Missing the inline ones is the most common failure:

```bash
# Review summaries + verdicts. Read the NEWEST, not the first — a push
# invalidates the prior review but leaves it here as DISMISSED, so
# .reviews[0] is often a stale verdict on an older commit. Every entry
# carries .commit.oid; check it against current HEAD before trusting it.
gh pr view <N> --repo <owner/repo> --json reviews \
  -q '.reviews[] | "[\(.submittedAt)] \(.author.login) \(.state) commit=\(.commit.oid[0:7])"'

# Inline (line-anchored) review comments — NOT in the above
gh api repos/<owner>/<repo>/pulls/<N>/comments \
  --jq '.[] | {path, line, user: .user.login, body}'

# Top-level PR conversation
gh pr view <N> --repo <owner/repo> --json comments

# Failing checks
gh pr checks <N> --repo <owner/repo>
```

## Step 4: triage each item

Sort every item into one of four buckets, and say which is which when you report:

| Bucket | Meaning | Action |
|---|---|---|
| **Blocking** | Real defect, or reviewer explicitly requested a change | Verify, then fix |
| **Non-blocking but right** | Correct observation the reviewer didn't block on | Usually fix — cheap and it's true |
| **Wrong** | The claim doesn't hold against the code | **Escalate to the user** |
| **Judgment call** | Correct but trades off against a deliberate decision | **Escalate to the user** |

An approval with a minor observation attached still deserves the fix if the observation is correct. "Non-blocking" describes the reviewer's stance, not whether it's true.

## Step 5: verify before you change anything

For each item, actually go read the code path it names. Confirm:

- The claim is true of the current code (not of an earlier revision, and not of code that only looks similar).
- The failure it describes can actually occur — trace the inputs. A reviewer flagging a branch as reachable is a claim to check, not a fact; if the state it describes is impossible, that's bucket 3 or a no-change with recorded reasoning.
- The fix doesn't undo something deliberate. Check `CLAUDE.md`, nearby comments, and the commit that introduced the line; a comment explaining *why* is a strong signal the reviewer missed context.

When you decline a change because the state it describes can't occur, give the user the reasoning explicitly — not just "not changing it". They're the one who'll defend it to the reviewer, so hand them the argument, not the conclusion.

## Step 6: make the changes

- One commit per coherent piece of feedback, message naming it as review follow-up.
- **Re-run the test suite after each change** (see below for this repo's command).
- Add or update a test whenever the feedback was about behavior — a fix without a test invites the same comment next time.
- Push normally. **Never force-push a branch someone else may have committed to**; if history needs rewriting, ask first.

## Step 7: report to the user — they post, not you

Give them one consolidated account, item by item, so they can act on the PR without re-reading it themselves:

- **Fixed** → what changed, and the commit SHA.
- **Declined** → the reasoning, in enough detail that they can post it as-is if they choose.
- **Question** → the answer.
- **Needs their decision** → see the escalation section below.

Only report something as fixed if it is actually fixed and verified. If a fix is partial, say which part.

If a reply to the reviewer looks warranted, offer the drafted text and let them decide whether to post it. Don't post it yourself, and don't resolve or dismiss anything on the PR.

## Escalating to the user

When you hit bucket 3 or 4, stop and report — briefly, with a recommendation, not a survey:

1. What the reviewer asked for (quote the relevant line).
2. Why you think it's wrong, or what the trade-off is.
3. What you'd do, and what it costs either way.

Then wait. Don't push a change to a disputed point before the user answers. If several items are disputed, batch them into one message rather than interrupting repeatedly.

## In this repo

- **Tests**: `uv run pytest` from the repo root — 105 tests, offline, a few
  seconds; no GPU, no network, no API key. Run it after every change and before
  every push, alongside `uv run ruff check .` and `uv run ruff format --check .`.
  Install deps first with `uv sync --group dev` (add `--extra vllm` only on the
  Linux GPU box). See `CLAUDE.md` for the invariants a change must not break.
- **Checks on every PR**: `test` (CI: pytest + ruff check + ruff format --check)
  and `review / claude-review` (the bot reviewer). The reviewer's check goes red
  when the bot posted no binary verdict, and a `needs-human-review` label means a
  human has been asked to review. Note that on this repo escalation is
  **label-only** — Deco354 is the sole maintainer and GitHub will not request
  review from a PR's own author, so nobody is notified. Check for the label.
- **Merge gate**: none. This is a private repo on a free plan, so branch
  protection is unavailable (`403: Upgrade to GitHub Pro or make this repository
  public`) and no check can be marked required. The bot is **advisory**: its
  approval gates nothing and its red check blocks nothing. Never merge on a bot
  approval alone; merge only when the user asks.
- **Review responses are posted by a human, never by an agent** — org-wide
  convention, and the reason for rule 1 above.
- **Claude-authored GitHub content must say so.** PR *descriptions* written by
  Claude need an explicit callout naming Claude as the author, e.g. a
  `> [!NOTE]` line at the top. (Comments don't arise — you don't post them.)

## Gotchas

- **`gh` and sandboxing.** If `gh` fails with a TLS/certificate error (e.g. `x509: OSStatus -26276`) or a spurious "token is invalid", that's the sandbox blocking the system trust store, not broken auth. Retry the same command with the sandbox disabled. Don't send the user off to re-authenticate over this.
- **New pushes can invalidate an approval.** After pushing follow-up commits, re-check `reviewDecision` rather than assuming the earlier approval still stands — a fix that addresses a review will typically dismiss that very review.
- **A re-review may be needed.** A bot reviewer triggered by pushes re-runs itself; a human who requested changes needs re-review requested explicitly.
- **Silence from CI is not success** — a check that never started looks the same as one that passed if you only test for absence of failure. Read the actual conclusions.
- **Working in a git worktree?** Run everything from the worktree directory, and never use bare `git stash`/`git stash pop` — the stash stack is shared with the main checkout and other worktrees.
