import os

import unittest

import numpy as np

from bevodevo.tests.policies.test_mlps import TestMLPPolicy

from bevodevo.policies.rnns import GatedRNNPolicy


class TestGatedRNNPolicy(TestMLPPolicy):

    def setUp(self):
        self.policy = GatedRNNPolicy(params=None)

if __name__ == "__main__": #pragma: no cover
    unittest.main(verbosity=2)
