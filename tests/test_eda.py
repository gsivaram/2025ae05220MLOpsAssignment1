from heartrisk import eda


def test_generate_all_writes_every_figure(clean_df, tmp_path):
    paths = eda.generate_all(clean_df, out_dir=tmp_path)
    assert len(paths) == len(eda.ALL_PLOTS)
    for p in paths:
        assert p.exists() and p.stat().st_size > 10_000, f"{p.name} missing or empty"
