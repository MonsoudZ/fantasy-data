

def test_fit_seasons_reaches_the_current_season():
    """Regression: this was hardcoded to [2023, 2024], so the feature frame
    stopped at 2024 and every later season became unprojectable -- project() got
    an empty test set and LightGBM raised "Input data must be 2 dimensional and
    non empty", killing the lineup optimizer and the props tab."""
    from ffdata.matchup import fit_seasons

    seasons, resid = fit_seasons(2019, through=2025)
    assert seasons[-1] == 2025, "must include the season we'll be asked to project"
    assert seasons[0] == 2019 and seasons == list(range(2019, 2026))
    assert resid == [2024, 2025]

    # And it keeps moving with the calendar rather than freezing again.
    later, _ = fit_seasons(2019, through=2030)
    assert later[-1] == 2030


def test_fit_seasons_honours_an_explicit_residual_range():
    from ffdata.matchup import fit_seasons

    seasons, resid = fit_seasons(2019, resid_seasons=[2021, 2022], through=2025)
    assert resid == [2021, 2022]
    # Still has to reach `through`, even when residuals stop earlier.
    assert seasons[-1] == 2025
