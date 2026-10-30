---
name: pickup-issue
description: Check a GitHub issue in this repo against current main before planning work on it. Use when asked to work on, implement, pick up or plan an issue.
argument-hint: [issue-number]
allowed-tools: Bash(gh api repos/{owner}/{repo}/issues/*) Bash(git log *) Bash(git diff *)
---

# Picking up an issue

## The issue and its comments

```!
gh api repos/{owner}/{repo}/issues/$ARGUMENTS --jq '"#\(.number) \(.title) [\(.state)]\n\n\(.body)"'
gh api repos/{owner}/{repo}/issues/$ARGUMENTS/comments --paginate --jq '.[] | "\n--- Comment by \(.user.login), \(.created_at)\n\n\(.body)"'
```

## Before planning

This issue was written by Claude partway through other work. Treat it as a draft, not
a spec. Comments can change the proposal, so read them as part of it.

1. Check every concrete claim (paths, functions, line numbers, flags, current
   behaviour) against main, and list what no longer holds. If the note at the top names
   the commit it was written against, start from
   `git diff <commit>..origin/main -- <files it names>`. In a shallow clone, run
   `git fetch --depth=200 origin main` first.
2. Check which linked issues and PRs have merged, closed or are still open, and how
   that changes the steps.
3. Resolve the "to check first" and "Assumed, not checked" items you can. List the
   decisions that are the author's: the issue's open decisions and options, and
   anything that spends money or touches credentials.

## Then

Report in this order: what no longer holds, what changed in linked work, decisions
needed, then the plan. Don't edit files until the plan is approved.
