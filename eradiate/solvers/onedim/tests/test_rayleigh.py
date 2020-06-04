# def test_onedimsolverdict(variant_scalar_mono):
#     # Construct
#     solver = OneDimSolverDict()
#     assert solver.mode == DEFAULT_ONEDIMDICT_MODE
#     assert solver.illumination == DEFAULT_ONEDIMDICT_ILLUMINATION
#     assert solver.measure == DEFAULT_ONEDIMDICT_MEASURE
#     assert solver.atmosphere == None
#     assert solver.surface == DEFAULT_ONEDIMDICT_SURFACE
#
#     # Run simulation with default parameters
#     assert solver.run() == 0.1591796875
#
#     # Run simulation with custom emitter
#     solver = OneDimSolverDict(illumination={'type': 'constant'})
#     assert np.isclose(solver.run(), 0.5, rtol=1e-2)

