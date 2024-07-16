import eradiate
from eradiate.test_tools.test_cases.atmospheres import (
    absorption_database_error_handler_config,
    create_rpv_afgl1986_brfpp,
    create_rpv_afgl1986_continental_brfpp,
)
from eradiate.test_tools.util import append_doc


class BenchmarkAtmosphere:
    error_handler_config = None

    def setup(self):
        eradiate.set_mode("ckd")
        self.error_handler_config = absorption_database_error_handler_config()

    @append_doc(create_rpv_afgl1986_continental_brfpp)
    def time_rpv_afgl1986_continental_brfpp(self):
        r"""
        RPV AFGL1986 Aerosol benchmark test
        ====================================

        This is a benchmark test, which records the time taken for the
        experiment to run. The test is done multiple times to get a s
        statistical result

        """

        exp = create_rpv_afgl1986_continental_brfpp(self.error_handler_config)
        eradiate.run(exp, spp=1000)

    @append_doc(create_rpv_afgl1986_brfpp)
    def time_rpv_afgl1986_brfpp(self):
        r"""
        RPV AFGL1986 benchmark test
        ===========================

        This is a benchmark test, which records the time taken for the
        experiment to run. The test is done multiple times to get a
        statistical result

        """
        exp = create_rpv_afgl1986_brfpp(self.error_handler_config)
        eradiate.run(exp, spp=1000)
