import eradiate
from eradiate.test_tools.test_cases.romc import (
    create_het06_brfpp,
    fetch_het06_brfpp,
)
from eradiate.test_tools.util import append_doc


class BenchmarkROMC:
    data = None

    def setup(self):
        eradiate.set_mode("ckd")
        self.data = fetch_het06_brfpp()

    @append_doc(create_het06_brfpp)
    def time_het06_brfpp(self):
        r"""
        Coniferous forest (HET06) benchmark test
        ========================================

        This is a benchmark test, which records the time taken for the
        experiment to run. The test is done multiple times to get a
        statistical result

        """

        exp = create_het06_brfpp(self.data)
        eradiate.run(exp)
   