from eradiate.thermoprops.afgl1986 import make_profile


def test_make_profile():
    # All thermophysical properties profile can be made
    model_ids = [
        "midlatitude_summer",
        "midlatitude_winter",
        "subarctic_summer",
        "subarctic_summer",
        "subarctic_winter",
        "us_standard",
    ]
    for model_id in model_ids:
        ds = make_profile(model_id=model_id)
