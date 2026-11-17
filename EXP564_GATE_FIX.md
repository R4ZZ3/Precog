# EXP564 schedule-parity gate correction

Source-only patch on `1e560bcf919eda557a3f84fa7fe35b9c4b1bc00e`, in
`ft_fork_564_gate` / branch `codex-exp564-summary`. No production code, kernel,
weights, native library, cache policy, scheduling or GPU-test input changes.
No GPU execution, serve or replay was performed while preparing this patch.
The user's decision on another run remains outstanding.

## Counter audit

The failure log reports token 8 with copied_bytes=663552 and published=48
in both schedules, but yields_to_demand=8 versus 0. The dictionary-equality
assertion was unsuitable for schedule telemetry. This observation does not
establish the determinism of the other transport fields.

| Summary field | Equality between schedules? | Code reason |
|---|---|---|
| yields_to_demand | No | `kernels.py:150-151`: counts CTAs observing BUSY at a chunk boundary |
| expired_calls | No | `kernels.py:146-149`: counts CTAs observing the target-layer deadline before finishing their cursor |
| copied_bytes | No | `kernels.py:145-178`: only stripes executed before yield/expiry contribute; partial work counts too |
| published | No | `kernels.py:179-186`: increments only when all chunk/stripe payload updates for an expert complete before the section stops |
| misses_from_stage | No | `striped_demand.py:41-55`: depends on observing matching IDS and READY generation at demand launch |
| misses_from_host | No individually | Complement of staged misses; host+stage must equal the deterministic demand-miss total |
| host_misses_while_pending | No | `striped_demand.py:43-57`: the winning source choice depends on publication visibility; pending is a subset of host misses |
| unscored_token_ids | Yes | Determined by supplied token IDs/table bounds; zero for this fixture |
| evict_events_scored | Yes | Determined by routes, valid scores, J and the identical cache event sequence |
| evict_target_batches_reranked | Yes | Determined by identical candidate sets and owner mapping |
| evict_slots_rescued | Yes | Determined by identical keep scores, candidates and clock-swap rule |
| evict_nonfinite_tokens | Yes | Determined by the same scoring inputs; zero for this fixture |

The files in the table are relative to `commitmoe_prefetch/commitmoe_prefetch`
except `striped_demand.py`, under `python/freetoken/moe`.

The pump's expiry branch precedes its BUSY branch. A CTA can count an expiry,
count a yield, or execute another stripe depending on when the other stream
advances. Joining the streams at the token end makes the results readable;
it does not finish or publish expired work. Consequently neither publication
count nor copied-byte count has a general exact-equality contract. Their
agreement in the failed run is consistent with all the fixture's selected
experts finishing, not proof of a universal invariant. Host/stage choices can
vary while both paths deliver the same expert bytes to the same cache slot.

## Patch contract

`test_split_equals_one_pass_at_every_event` still compares every event's
hit bits, victim slots, source expert IDs, slot maps, ages, clock and learned
counters exactly. Token keep scores, prefetch probabilities and validity are
also exact. Both runs retain their existing resident-row byte checks. The
existing non-vacuity checks for victims, learned reranks and stage consumption
are retained. The distinct host-versus-stage transport choice is not what
the event probe's `src_indices` represents.

The new pure-CPU helper classifies every summary key explicitly and fails on
unknown/missing keys. For each request prefix and schedule it requires:

- Integer, nonnegative counters; zero nonfinite scores and invalid token IDs.
- Stage+host misses equal the cumulative missing hit bits recorded at all
  layer events; pending-host misses do not exceed host misses.
- Staged misses do not exceed publications; publications do not exceed the
  per-section caps summed over the eligible tokens.
- Published experts' complete bytes do not exceed copied bytes; copied
  bytes do not exceed the same cap budget times the expert's row size;
  copied bytes are aligned to the kernel's eight-byte words.
- Every cumulative counter is monotonic within a request, with the comparison
  reset at request boundaries; deterministic summary fields compare exactly.

All seven transport fields are retained as a `prefetch_transport_counters`
pytest property for both schedules and every request/token prefix. To retain
that property as a durable artifact in a future suite, use
`--junitxml=RUN/gpu_tests_BUNDLE.xml -o junit_family=legacy` and bind the XML
alongside the corresponding suite log in the new execution manifest. This
is diagnostic output, not a replacement for a numerical/decision gate.

The existing chosen-score/sleep and no-pump fixtures in
`test_gpu_schedule.py` still assert their scenario-specific publication and
host-fallback expectations. They are not proofs that transport counts must
agree between arbitrary schedules. This patch does not change those tests
or claim that every possible GPU-test timing issue has been eliminated.

## CPU validation only

From commitmoe_export:

```bash
rtk proxy env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
python -m pytest -q \
--noconftest -p no:cacheprovider \
ft_fork_564_gate/commitmoe_prefetch/tests/test_summary_contract.py
```

22 passed in 0.18 s. Tests import no torch or runtime module. They reproduce
the old equality failure on an 8-versus-0 yield fixture, allow all seven
transport fields to differ with valid accounting, reject changed deterministic
counters, and reject bad miss sums, partial-publication accounting, excess
copy budgets, non-monotonic counters, nonfinite counts and unclassified fields.
The actual GPU gate has not been run. The failed attempt remains failed.

## Re-pin checklist before any user-approved new attempt

1. Review and adopt the test patch into the serving fork; use that fork's
   actual resulting commit. Keeping the existing `ft_fork_ov` path minimizes
   unrelated changes. This source worktree has no prepared ignored native
   libraries and is not presented as a ready serving installation.
2. Set the new commit in `exp564_run.FORK_COMMIT` and the chain's `COMMIT`.
   Set `exp564_analyze.FORK_COMMIT` locally instead of inheriting the old
   value from the frozen EXP554r helper. Its existing metadata checks will
   then require the new commit and a clean fork. Do not rewrite historical
   metadata or loosen commit/path checks to allow both old and new.
3. If the path remains `ft_fork_ov`, no runtime/native file was changed by
   this test patch: verify the same seven native-library and ten release-file
   hashes. No native rebuild or new hashes should be needed. Keep the old
   input manifest immutable when its bytes/paths remain valid. If a new fork
   path is chosen, explicitly update runner `r.CONFIG['fork_split']`, chain
   FORK, analyzer default and its literal preflight path, and create a new
   input manifest for the reviewed native set at those paths. Use identical
   reviewed library bytes rather than assuming source equivalence; pin its
   new digest locally in the analyzer, not in historical helpers.
4. Add all five GPU-test sources and the new `summary_contract.py` to the
   execution manifest (and CPU regressions if run). Import discovery in the
   runner/analyzer does not discover modules imported by the separate pytest
   process. Preserve historical outcome/text pins and frozen helper pins;
   their evidence is unchanged by this test-only patch.
5. The same five explicitly named GPU-test files still collect 77 tests. The
   version-1 applicability skip stays at test_gpu_learned_evict.py:307, so R
   remains exactly 75 passed/2 identified skips, C 77 passed/0 skipped. The
   new 22-case CPU file is a separate invocation, not part of those counts.
   Any other failure stops; no repeat of an unchanged failed gate.
6. Preregister the corrected comparison contract before the new suites,
   obtain the user's run decision/claim/window, use fresh run and analysis
   folders, and retain both stopped attempts. Fresh confirms and balanced
   blocks retain the old per-arm miss/text identity, memory, telemetry and
   bound manual-state gates. No old timing enters the estimate. A future
   new-gate GPU failure still stops the attempt without retrying to pass.
