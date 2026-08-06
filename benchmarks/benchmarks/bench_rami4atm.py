import eradiate
from eradiate.test_tools.test_cases.rami4atm import CASES, create_toa
from eradiate.test_tools.util import append_doc

#: Case timed by this benchmark
CASE_ID = "hom00_bla_sd2s_m03_z30a000_brfpp"

#: Sample count, matching the regression test suite
SPP = 1000


class BenchmarkRami4ATM:
    experiments = None

    def setup(self):
        eradiate.set_mode("ckd")
        self.experiments = CASES[CASE_ID].make_experiments(spp=SPP)

    @append_doc(create_toa)
    def time_rami4atm_hom00_bla_sd2s_m03_z30a000_brfpp(self):
        r"""
        RAMI4ATM HOM00_BLA_SD2S_M03 benchmark test
        ==========================================

        This is a benchmark test, which records the time taken for the
        experiment to run. The test is done multiple times to get a
        statistical result

        """

        for exp in self.experiments:
            eradiate.run(exp)
