"""CPU assertions for schedule parity without assuming identical prefetch timing."""

DECISION_FIELDS = (
    'unscored_token_ids',
    'evict_events_scored',
    'evict_target_batches_reranked',
    'evict_slots_rescued',
    'evict_nonfinite_tokens',
)
TRANSPORT_FIELDS = (
    'copied_bytes',
    'yields_to_demand',
    'expired_calls',
    'published',
    'misses_from_stage',
    'misses_from_host',
    'host_misses_while_pending',
)
SUMMARY_FIELDS = frozenset(DECISION_FIELDS + TRANSPORT_FIELDS)


def check_summary(summary, *, misses, row_bytes, copy_budget, previous=None):
    """Check one request prefix; copy_budget counts eligible expert publications."""
    assert set(summary) == SUMMARY_FIELDS, ('unclassified summary fields', set(summary) ^ SUMMARY_FIELDS)
    assert all(type(value) is int and value >= 0 for value in summary.values()), summary
    assert type(misses) is int and misses >= 0
    assert type(row_bytes) is int and row_bytes > 0 and row_bytes % 8 == 0
    assert type(copy_budget) is int and copy_budget >= 0
    assert summary['unscored_token_ids'] == summary['evict_nonfinite_tokens'] == 0, summary
    assert summary['misses_from_stage'] + summary['misses_from_host'] == misses, summary
    assert summary['host_misses_while_pending'] <= summary['misses_from_host'], summary
    assert summary['misses_from_stage'] <= summary['published'] <= copy_budget, summary
    assert summary['published'] * row_bytes <= summary['copied_bytes'] <= copy_budget * row_bytes, summary
    assert summary['copied_bytes'] % 8 == 0, summary
    if previous is not None:
        assert set(previous) == SUMMARY_FIELDS
        assert all(summary[name] >= previous[name] for name in SUMMARY_FIELDS), (previous, summary)


def compare_summaries(one_pass, split, *, misses, row_bytes, copy_budget, previous=None):
    """Compare decisions exactly; return timing-sensitive counters for diagnostics."""
    for index, summary in enumerate((one_pass, split)):
        check_summary(summary, misses=misses, row_bytes=row_bytes, copy_budget=copy_budget,
                      previous=None if previous is None else previous[index])
    assert {name: one_pass[name] for name in DECISION_FIELDS} == {
        name: split[name] for name in DECISION_FIELDS
    }, (one_pass, split)
    return {arm: {name: summary[name] for name in TRANSPORT_FIELDS}
            for arm, summary in (('one_pass', one_pass), ('split', split))}
