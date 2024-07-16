import eradiate
from eradiate.test_tools.test_cases.rami4atm import (
    create_rami4atm_hom00_bla_sd2s_m03_z30a000_brfpp,
)
from eradiate.test_tools.util import append_doc


class BenchmarkRami4ATM:
    def setup(self):
        eradiate.set_mode("ckd")

    @append_doc(create_rami4atm_hom00_bla_sd2s_m03_z30a000_brfpp)
    def time_rami4atm_hom00_bla_sd2s_m03_z30a000_brfpp(self):
        r"""
        RAMI4ATM HOM00_BLA_SD2S_M03 benchmark test
        ==========================================

        This is a benchmark test, which records the time taken for the
        experiment to run. The test is done multiple times to get a
        statistical result

        """

        exp = create_rami4atm_hom00_bla_sd2s_m03_z30a000_brfpp()
        eradiate.run(exp)
