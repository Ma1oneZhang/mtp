# MTP training experiments

The default draft architecture combines two additions to DFlash:

- a two-tap, block-local dynamic mixer around attention and MLP in every draft
  layer;
- a rank-256 vanilla Markov head that adds a low-rank transition bias from the
  previous token to the parallel backbone logits.

The Markov successor matrix starts at zero, so the default architecture has the
same initial logits as the two-tap control. Training uses teacher predecessors
for token CE. Greedy acceptance follows the inference path and feeds each
predicted token into the next Markov transition.

Matrix parameters use Muon and auxiliary parameters use AdamW. The default loss
is equal-weight hard-label token CE. Pass `--experiment baseline` or
`--experiment two_tap` for controlled ablations. `pal10` remains an explicit
loss option in the distributed trainer.

In the paired seed-42, four-GPU CE/Muon run used to select the default, the
100-step means at synchronized step 3,470 were:

| Architecture | Plain CE | Sequential greedy acceptance | Median step |
| --- | ---: | ---: | ---: |
| two-tap | 4.4918 | 2.4555 | 268 ms |
| two-tap + Markov | 3.2283 | 2.6367 | 278 ms |

These are training-batch measurements from an unfinished 10,000-step run. The
acceptance number follows the self-generated Markov path; it does not use the
teacher predecessor used by CE. A fixed holdout evaluation remains required
for a final model-quality claim.

The distributed entrypoint starts one synchronized job across the requested
number of GPUs. Each run directory contains `run.json`, streamed JSON logs,
atomic checkpoints, and `completed.json`. Gradient norms separate convolution,
Markov, fusion, selector, and backbone parameter families.

```bash
torchrun --nproc-per-node=8 train_distributed.py \
  --target-model <target-model> \
  --draft-model <draft-config> \
  --data <parquet-directory> \
  --output-dir <experiment-run-directory> \
  --storage-root <experiment-storage-root> \
  --expected-world-size 8
```

The trainers sample actual stored-parameter changes at step 1 and every 100
steps by default. Sampled JSON records include parameter and update norms,
relative updates, effective step sizes, gradient/update cosine, changed-element
fractions, and update energy fractions. Markov parameters have their own metric
family. Set `--update-log-interval 0` only when measuring observer overhead.
