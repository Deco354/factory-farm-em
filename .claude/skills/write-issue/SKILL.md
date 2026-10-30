---
name: write-issue
description: Write and open a GitHub issue in this repo in its house format, recording the commit it was written against. Use when the user asks for an issue, or when another command (such as subtask) hands an issue off to this skill. Don't open issues unprompted.
argument-hint: [what the issue is about]
allowed-tools: Bash(git merge-base *) Bash(git rev-parse *) Bash(git branch *)
---

# Writing an issue

## Where it is written from

- Commit on main: !`git merge-base HEAD origin/main 2>/dev/null || git rev-parse HEAD`
- Branch: !`git branch --show-current`

The commit above is where this branch left main. Record it rather than the branch's
own HEAD: branch commits don't reach main after a squash merge.

## Format

Open with the authorship note and the commit:

```
> [!NOTE]
> Written by Claude Code at the author's direction. Written against `<commit>` on main, while working on #<PR> (if any).
```

Then these sections, in this order. Leave out a section only where it says so.

- **Problem** (or **Goal** for new work): what's wrong or wanted, and where it came from.
- **Proposal**: the change. If it isn't decided, give the options and put the choice in Open decisions.
- **Checked**: the facts the issue rests on that you verified in this session, each with where (file and function, or line). Only what you actually checked.
- **Assumed, not checked**: what the proposal relies on that nobody verified, including "to check first" items. Write "None" if there are none.
- **Open decisions**: choices that are the author's, not the implementer's: options left open, and anything that spends money or touches credentials. Write "None" if there are none.
- **Done when**: outcomes someone can observe.
- **Related**: linked issues and PRs, with how each bears on this one. Leave out if none.

## Rules

- It must stand on its own. Whoever picks it up starts a fresh session with none of
  this conversation, so restate decisions and context; don't point at the chat.
- Follow `CLAUDE.md`'s testing rules when describing tests.
- Date anything that changes outside the repo: prices, box state, upstream versions
  ("on 2026-10-30").
- The repo is public: no secret values or token contents.
- Match the existing issues' style: plain sentences, concrete paths and commands.

## Opening it

Write the body to a file in your scratchpad, then run
`gh issue create --title "<title>" --body-file <file>`. If `gh` says GraphQL isn't
available (cloud sessions), use
`gh api repos/{owner}/{repo}/issues -f title="<title>" -F body=@<file>` instead.
Reply with the issue's link.
