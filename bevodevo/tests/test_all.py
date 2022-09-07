import unittest

from bevodevo.tests.policies.test_mlps import TestMLPPolicy,\
    TestHebbianMLP,\
    TestHebbianCAMLP,\
    TestHebbianCAMLP2,\
    TestCPPNHebbianMLP,\
    TestCPPNMLPPolicy,\
    TestABCHebbianMLP

if __name__ == "__main__": #pragma: no cover
    unittest.main(verbosity=2)
