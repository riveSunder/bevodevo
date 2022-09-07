import os

import unittest

import numpy as np

from bevodevo.tests.policies.test_mlps import TestMLPPolicy

from bevodevo.policies.cnns import ImpalaCNNPolicy


class TestImpalaCNNPolicy(unittest.TestCase):

    def setUp(self):
        self.policy = ImpalaCNNPolicy(params=None)

    def test_set_params(self):
        
        my_params = self.policy.get_params()

        new_params = np.random.randn(*my_params.shape)

        self.policy.set_params(new_params)
        recovered_params = self.policy.get_params()

        # some precision is lost going back and forth
        # between numpy and torch here, therefore 1e-6
        
        self.assertGreater(1e-6, np.abs(new_params - recovered_params).max())

        self.policy.set_params(my_params)
        recovered_params = self.policy.get_params()

        self.assertNotIn(False, my_params == recovered_params)

if __name__ == "__main__": #pragma: no cover
    unittest.main(verbosity=2)
