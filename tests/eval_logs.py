"""Hand-written `.eval` logs with the real structure, for tests that read a log directory
(`fc export`, and what `fc run` does after eval_set)."""

from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalSample, EvalSpec, write_eval_log


def write_log(
    path,
    task_id,
    created,
    status,
    sample_ids,
    *,
    task="plumbing_task",
    model="mockllm/model",
    metadata=None,
    samples=None,
):
    """One log of `task` on `model`. `samples` (EvalSamples) replace the bare ones made
    from `sample_ids`; `metadata` is the eval's, as `fc run` sets it (`models`, ...)."""
    samples = samples or [EvalSample(id=s, epoch=1, input="q", target="") for s in sample_ids]
    log = EvalLog(
        eval=EvalSpec(
            created=created,
            task=task,
            task_id=task_id,
            dataset=EvalDataset(samples=len(samples), sample_ids=[s.id for s in samples]),
            model=model,
            config=EvalConfig(),
            metadata=metadata,
        ),
        status=status,
        samples=samples,
    )
    write_eval_log(log, str(path))
