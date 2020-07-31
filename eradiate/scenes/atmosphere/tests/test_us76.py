import pytest
from eradiate.scenes.atmosphere.us76 import *


def test_create():
    variables = [
        "pressure",
        "temperature",
        "number_density",
        "total_number_density",
    ]
    z = Q_(np.linspace(0.0, 100000.0, 101), "meter")
    ds = create(z, variables=variables)

    dims = ds.dims
    assert len(dims) == 2
    assert "altitude" in dims
    assert "species" in dims

    coords = ds.coords
    assert len(coords) == 2
    assert (coords["altitude"] == z.magnitude).all()
    assert [s for s in coords["species"]] == [s for s in SPECIES]

    for var in variables:
        assert var in ds

    assert all(
        [
            x in ds.attrs
            for x in ["convention", "title", "history", "source", "references"]
        ]
    )


def test_compute_below_86_km_layers_boundary_altitudes():
    """
    We test the computation of the atmospheric variables (pressure,
    temperature and mass density) at the level altitudes, i.e. at the model
    layer boundaries. We assert correctness by comparing their values with the
    values from the table 1 of the U.S. Standard Atmosphere 1976 document.
    """

    z = to_altitude(np.array(H))
    ds = create(
        z, variables=["pressure", "temperature", "mass_density"]
    )

    level_temperature = np.array(
        [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.87])
    level_pressure = np.array(
        [101325.0, 22632.0, 5474.8, 868.01, 110.90, 66.938, 3.9564, 0.37338])
    level_mass_density = np.array(
        [
            1.225,
            0.36392,
            0.088035,
            0.013225,
            0.0014275,
            0.00086160,
            0.000064261,
            0.000006958,
        ]
    )

    assert np.allclose(ds["temperature"].values, level_temperature, rtol=1e-4)
    assert np.allclose(ds["pressure"].values, level_pressure, rtol=1e-4)
    assert np.allclose(ds["mass_density"].values, level_mass_density, rtol=1e-3)


def test_compute_below_86_km_arbitrary_altitudes():
    """
    We test the computation of the atmospheric variables (pressure,
    temperature and mass density) at arbitrary altitudes. We assert correctness
    by comparing their values to the values from table 1 of the U.S. Standard
    Atmosphere 1976 document.
    """

    # The values below were selected arbitrarily from Table 1 of the document
    # such that there is at least one value in each of the 7 temperature
    # regions.
    h = np.array([
        200.0,
        1450.0,
        5250.0,
        6500.0,
        9800.0,
        17900.0,
        24800.0,
        27100.0,
        37200.0,
        40000.0,
        49400.0,
        61500.0,
        79500.0,
        84000.0,
    ])
    temperatures = np.array([
        286.850,
        278.725,
        254.025,
        245.900,
        224.450,
        216.650,
        221.450,
        223.750,
        243.210,
        251.050,
        270.650,
        241.250,
        197.650,
        188.650,
    ])
    pressures = np.array([
        98945.0,
        85076.0,
        52239.0,
        44034.0,
        27255.0,
        7624.1,
        2589.6,
        1819.4,
        408.7,
        277.52,
        81.919,
        16.456,
        0.96649,
        0.43598,
    ])
    mass_densities = np.array([
        1.2017,
        1.0633,
        0.71641,
        0.62384,
        0.42304,
        0.12259,
        0.040739,
        0.028328,
        0.0058542,
        0.0038510,
        0.0010544,
        0.00023764,
        0.000017035,
        0.0000080510,
    ])

    z = to_altitude(h)
    ds = create(
        z, variables=["temperature", "pressure", "mass_density"]
    )

    assert np.allclose(temperatures, ds["temperature"].values, rtol=1e-4)
    assert np.allclose(pressures, ds["pressure"].values, rtol=1e-4)
    assert np.allclose(mass_densities, ds["mass_density"].values, rtol=1e-4)


def test_init_data_set():

    def check_data_set(ds):
        for var in VARIABLES:
            assert var in ds
            assert np.isnan(ds[var].values).all()

        assert ds["number_density"].values.ndim == 2
        assert all(ds["species"].values == ['N2', 'O2', 'Ar', 'CO2', 'Ne', 'He', 'Kr', 'Xe', 'CH4', 'H2', 'O', 'H'])

    z1 = Q_(np.linspace(0., 50000.), "m")
    ds1 = init_data_set(z1)
    check_data_set(ds1)

    z2 = Q_(np.linspace(120000., 650000.), "m")
    ds2 = init_data_set(z2)
    check_data_set(ds2)

    z3 = Q_(np.linspace(70000., 100000.), "m")
    ds3 = init_data_set(z3)
    check_data_set(ds3)


def test_compute_levels_temperature_and_pressure_low_altitude():
    tb, pb = compute_levels_temperature_and_pressure_low_altitude()

    level_temperature = np.array(
        [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.87])
    level_pressure = np.array(
        [101325.0, 22632.0, 5474.8, 868.01, 110.90, 66.938, 3.9564, 0.37338])

    assert np.allclose(tb, level_temperature, rtol=1e-3)
    assert np.allclose(pb, level_pressure, rtol=1e-3)


def rtol(v, ref):
    return np.abs(v - ref) / ref


def test_compute_number_density():
    # the following altitudes values are chosen arbitrarily
    altitudes = Q_(
        np.array([
            86.0,
            90.0,
            95.0,
            100.0,
            110.0,
            120.0,
            150.0,
            200.0,
            300.0,
            400.0,
            500.0,
            600.0,
            700.0,
            800.0,
            900.0,
            1000.0,
        ]),
        "km",
    )
    mask = altitudes.magnitude > 150.0

    # the corresponding number density values are from NASA (1976) - U.S.
    # Standard Atmosphere, table VIII (p. 210-215)
    values = {
        "N2": np.array([
            1.13e20,
            5.547e19,
            2.268e19,
            9.210e18,
            1.641e18,
            3.726e17,
            3.124e16,
            2.925e15,
            9.593e13,
            4.669e12,
            2.592e11,
            1.575e10,
            1.038e9,
            7.377e7,
            5.641e6,
            4.626e5,
        ]),
        "O": np.array([
            O_7,
            2.443e17,
            4.365e17,
            4.298e17,
            2.303e17,
            9.275e16,
            1.780e16,
            4.050e15,
            5.443e14,
            9.584e13,
            1.836e13,
            3.707e12,
            7.840e11,
            1.732e11,
            3.989e10,
            9.562e9,
        ]),
        "O2": np.array([
            O2_7,
            1.479e19,
            5.83e18,
            2.151e18,
            2.621e17,
            4.395e16,
            2.750e15,
            1.918e14,
            3.942e12,
            1.252e11,
            4.607e9,
            1.880e8,
            8.410e6,
            4.105e5,
            2.177e4,
            1.251e3,
        ]),
        "Ar": np.array([
            AR_7,
            6.574e17,
            2.583e17,
            9.501e16,
            1.046e16,
            1.366e15,
            5.0e13,
            1.938e12,
            1.568e10,
            2.124e8,
            3.445e6,
            6.351e4,
            1.313e3,
            3.027e1,
            7.741e-1,
            2.188e-2,
        ]),
        "He": np.array([
            7.582e14,
            3.976e14,
            1.973e14,
            1.133e14,
            5.821e13,
            3.888e13,
            2.106e13,
            1.310e13,
            7.566e12,
            4.868e12,
            3.215e12,
            2.154e12,
            1.461e12,
            1.001e12,
            6.933e11,
            4.850e11,
        ]),
        "H": np.array([
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            3.767e11,
            1.630e11,
            1.049e11,
            8.960e10,
            8.0e10,
            7.231e10,
            6.556e10,
            5.961e10,
            5.434e10,
            4.967e10,
        ]),
    }

    n = compute_number_densities_high_altitude(altitudes)

    # print('N2:', rtol(n[0], values['N2']))
    assert np.allclose(n["N2"], values["N2"], rtol=0.01)
    # print('O:',  rtol(n[1], values['O']))
    assert np.allclose(n["O"], values["O"], rtol=0.1)
    # print('O2:', rtol(n[2], values['O2']))
    assert np.allclose(n["O2"], values["O2"], rtol=0.01)
    # print('Ar:', rtol(n[3], values['Ar']))
    assert np.allclose(n["Ar"], values["Ar"], rtol=0.01)
    # print('He:', rtol(n[4], values['He']))
    assert np.allclose(n["He"], values["He"], rtol=0.01)
    # print('H:', rtol(n[5][mask], values['H'][mask]))
    assert np.allclose(n["H"][mask], values["H"][mask], rtol=0.01)


def test_compute_mean_molar_mass():
    # test call with scalar altitude
    assert compute_mean_molar_mass_high_altitude(90.0) == M0
    assert compute_mean_molar_mass_high_altitude(200.0) == M["N2"]

    # test call with array of altitudes
    altitude = np.linspace(86, 1000, 915)
    assert np.allclose(
        compute_mean_molar_mass_high_altitude(altitude), np.where(altitude <= 100.0, M0, M["N2"])
    )


def test_compute_temperature_above_86_km():
    # test altitudes out of range raises value error
    with pytest.raises(ValueError):
        compute_temperature_high_altitude(10.0)

    # test call with scalar altitude
    assert np.isclose(compute_temperature_high_altitude(90.0), 186.87, rtol=1e-3)

    # test call with array of altitudes
    z = [100, 110, 120, 130, 200, 500]  # km
    assert np.allclose(
        compute_temperature_high_altitude(z),
        np.array([195.08, 240.00, 360.0, 469.27, 854.56, 999.24]),
        rtol=1e-3,
    )


def test_integrate():
    x = [0., 1., 3.]
    y = [4., 12., -2.]
    assert np.allclose(integrate(y, x), [0., 8., 18.])
