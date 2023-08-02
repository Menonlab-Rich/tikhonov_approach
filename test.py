import unittest
from tikhonov import Tikhonov
import numpy as np

class TestTikhonov(unittest.TestCase):
    def setUp(self):
        # Set up a simple test problem
        self.A = np.array([[1, 2], [3, 4]])
        self.Gamma = np.eye(2)
        self.alpha = 0.01
        self.tikhonov = Tikhonov(self.A, self.Gamma, self.alpha)

    def test_gradient_descent(self):
        b = np.array([1, 1])
        S0 = np.array([0, 0])
        learning_rate = 0.01
        num_iters = 990

        # Solve the problem using gradient descent
        S = self.tikhonov.gradient_descent(b, S0, learning_rate, num_iters)

        # Check that the solution is close to the expected solution
        expected_S = np.linalg.inv(self.A.T @ self.A + self.alpha * self.Gamma.T @ self.Gamma) @ self.A.T @ b
        self.assertTrue(np.allclose(S, expected_S, rtol=1e-3, atol=1e-3))

if __name__ == "__main__":
    unittest.main()
