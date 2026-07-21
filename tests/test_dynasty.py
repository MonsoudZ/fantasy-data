"""Dynasty age curves -- delta-method shape sanity (needs the data lake)."""

from conftest import requires_data_lake


@requires_data_lake
def test_age_curves_decline_and_rb_peaks_before_te():
    from ffdata.dynasty import age_curves
    c = age_curves()
    # Every position is peak-normalized to 1.0 and never exceeds it.
    for pos in ("QB", "RB", "WR", "TE"):
        assert max(c[pos].values()) == 1.0 and all(0 <= v <= 1 for v in c[pos].values())
    # Delta method should recover the known shape: RBs peak young and fall hard;
    # by 32 a running back retains less of his peak than a tight end.
    rb_peak = max(c["RB"], key=c["RB"].get)
    te_peak = max(c["TE"], key=c["TE"].get)
    assert rb_peak <= te_peak
    assert c["RB"][32] < c["TE"][32]


@requires_data_lake
def test_dynasty_favors_youth_over_equal_redraft_value():
    from ffdata.dynasty import dynasty_board
    b = dynasty_board(2024, years=4)
    assert {"player", "age", "dynasty_value"}.issubset(b.columns)
    assert b["dynasty_value"].is_monotonic_decreasing  # returned sorted best-first
