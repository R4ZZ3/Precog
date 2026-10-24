"""Synthetic CPU regressions; no torch, CUDA or runtime imports."""
import copy

import pytest

from summary_contract import DECISION_FIELDS, SUMMARY_FIELDS, compare_summaries


def summary():
    return dict(copied_bytes=64, yields_to_demand=8, expired_calls=0, published=2,
                misses_from_stage=1, misses_from_host=9, host_misses_while_pending=0,
                unscored_token_ids=0, evict_events_scored=4, evict_target_batches_reranked=5,
                evict_slots_rescued=3, evict_nonfinite_tokens=0)


def compare(a, b, **kwargs):
    return compare_summaries(a, b, misses=10, row_bytes=32, copy_budget=3, **kwargs)


def test_observed_yield_only_difference_is_allowed():
    a, b = summary(), summary()
    for value in (a, b):
        value.update(copied_bytes=663552, published=48)
    b['yields_to_demand'] = 0
    assert a != b  # The former full-dictionary equality fails on this fixture.
    result = compare_summaries(a, b, misses=10, row_bytes=13824, copy_budget=48)
    assert result['one_pass']['yields_to_demand'] == 8
    assert result['split']['yields_to_demand'] == 0


def test_all_transport_fields_can_change_without_changing_decisions():
    a, b = summary(), summary()
    b.update(copied_bytes=40, yields_to_demand=0, expired_calls=2, published=1,
             misses_from_stage=0, misses_from_host=10, host_misses_while_pending=2)
    result = compare(a, b)
    assert all(result['one_pass'][name] != result['split'][name] for name in result['one_pass'])


@pytest.mark.parametrize('field', DECISION_FIELDS)
def test_every_decision_counter_remains_a_gate(field):
    a, b = summary(), summary()
    b[field] += 1
    with pytest.raises(AssertionError):
        compare(a, b)


@pytest.mark.parametrize('field,value', [
    ('misses_from_host', 8), ('host_misses_while_pending', 10),
    ('published', 0), ('copied_bytes', 56), ('copied_bytes', 104),
    ('copied_bytes', 65), ('yields_to_demand', -1), ('expired_calls', True),
    ('published', 4), ('unscored_token_ids', 1), ('evict_nonfinite_tokens', 1),
])
def test_inconsistent_or_invalid_counters_fail(field, value):
    a = summary()
    a[field] = value
    with pytest.raises(AssertionError):
        compare(a, copy.deepcopy(a))


@pytest.mark.parametrize('change', ['missing', 'new'])
def test_schema_changes_require_explicit_classification(change):
    a = summary()
    if change == 'missing':
        a.pop('published')
    else:
        a['new_counter'] = 0
    with pytest.raises(AssertionError):
        compare(a, copy.deepcopy(a))


def test_counts_cannot_decrease_within_a_request():
    a, b = summary(), summary()
    previous = [copy.deepcopy(a), copy.deepcopy(b)]
    previous[0]['yields_to_demand'] = 9
    with pytest.raises(AssertionError):
        compare(a, b, previous=previous)
    assert set(a) == SUMMARY_FIELDS


def test_both_arms_nonfinite_does_not_pass_via_equal_bad_counts():
    a, b = summary(), summary()
    a['evict_nonfinite_tokens'] = b['evict_nonfinite_tokens'] = 1
    with pytest.raises(AssertionError):
        compare(a, b)
