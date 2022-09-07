import os

import unittest

import numpy as np

from bevodevo.policies.mlps import MLPPolicy, HebbianMLP,  ABCHebbianMLP

class TestMLPPolicy(unittest.TestCase):

    def setUp(self):
        self.policy = MLPPolicy(params=None)

    def test_mlp_forward(self):

        x = np.random.randn(1, self.policy.input_dim)
        
        output = self.policy(x)

        self.assertEqual(output.shape[-1], self.policy.action_dim)

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


class TestHebbianMLP(TestMLPPolicy):

    def setUp(self):
        self.policy = HebbianMLP(params=None)

class TestABCHebbianMLP(TestMLPPolicy):

    def setUp(self):
        self.policy = ABCHebbianMLP(params=None)

    def test_mlp_forward(self):

        x = np.random.randn(1, self.policy.input_dim)
        
        output = self.policy(x)

        self.assertEqual(output.shape[-1], self.policy.action_dim)

if __name__ == "__main__": #pragma: no cover
    unittest.main(verbosity=2)
        
