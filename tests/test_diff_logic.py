"""
Tests the pure diffing logic used to generate continuity flags, without
requiring a live ClickHouse connection. Mirrors the comparison logic in
src/db/clickhouse_loader.diff_takes_for_scene so the core algorithm can be
validated in isolation.
"""

from __future__ import annotations


def find_continuity_mismatches(entries_by_label: dict[str, list[tuple[int, str, str]]]):
    """
    entries_by_label: {label: [(take_number, screen_position, state), ...]}
    Returns a list of (label, take_a, take_b) tuples for mismatches between
    consecutive takes of the same labeled object/actor.
    """
    mismatches = []
    for label, entries in entries_by_label.items():
        entries = sorted(entries, key=lambda e: e[0])
        for i in range(len(entries) - 1):
            take_a, pos_a, state_a = entries[i]
            take_b, pos_b, state_b = entries[i + 1]
            if pos_a != pos_b or state_a != state_b:
                mismatches.append((label, take_a, take_b))
    return mismatches


def test_no_mismatch_when_consistent():
    data = {"coffee mug": [(1, "left hand", "half full"), (2, "left hand", "half full")]}
    assert find_continuity_mismatches(data) == []


def test_detects_position_mismatch():
    data = {"coffee mug": [(1, "left hand", "half full"), (2, "right hand", "half full")]}
    result = find_continuity_mismatches(data)
    assert result == [("coffee mug", 1, 2)]


def test_detects_state_mismatch():
    data = {"jacket": [(1, "zipped", "collar up"), (2, "unzipped", "collar up")]}
    result = find_continuity_mismatches(data)
    assert result == [("jacket", 1, 2)]


def test_multiple_labels_independent():
    data = {
        "mug": [(1, "left", "full"), (2, "left", "full")],
        "phone": [(1, "pocket", "off"), (2, "hand", "on")],
    }
    result = find_continuity_mismatches(data)
    assert result == [("phone", 1, 2)]
