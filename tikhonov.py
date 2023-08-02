import torch # PyTorch is used for gradient descent
import numpy as np

class Tikhonov:
    def __init__(self, A, Gamma, alpha) -> None:
        '''
        Solve the Tikhonov regularized least squares problem using different methods.
        
        Parameters
        ----------
        A : np.ndarray
            The design matrix
        Gamma : np.ndarray
            The Tikhonov matrix
        alpha : float
            The regularization parameter
        
        '''
        self.A = A
        self.Gamma = Gamma
        self.alpha = alpha

    def gradient_descent(self, b, S0, learning_rate, num_iters):
        '''
        Solve the Tikhonov regularized least squares problem using gradient descent.
        
        Parameters
        ----------
        b : np.ndarray
            The response vector
        S0 : np.ndarray
            The initial guess for the solution
        learning_rate : float
            The learning rate for gradient descent
        num_iters : int
            The number of iterations to perform
        
        Returns
        -------
        np.ndarray
            The solution to the problem
        '''
        # Convert the inputs to PyTorch tensors
        b = torch.tensor(b, dtype=torch.float32, requires_grad=False)
        S = torch.tensor(S0, dtype=torch.float32, requires_grad=True)
        A = torch.tensor(self.A, dtype=torch.float32, requires_grad=False)
        Gamma = torch.tensor(self.Gamma, dtype=torch.float32, requires_grad=False)
        # Create an Adam optimizer
        optimizer = torch.optim.Adam([S], lr=learning_rate)

        for _ in range(int(num_iters)):  # make sure num_iters is an integer
            # Zero the gradients
            optimizer.zero_grad()
            # Compute the cost function
            cost = torch.norm(torch.matmul(A, S) - b)**2 + torch.norm(torch.matmul(self.alpha * Gamma, S))**2

            # Compute the gradients
            cost.backward()

            # Update S
            optimizer.step()

        return S.detach().numpy()
    
    def direct(self, b):
        '''
        Solve the Tikhonov regularized least squares problem directly.
        
        Parameters
        ----------
        b : np.ndarray
            The response vector
        
        Returns
        -------
        np.ndarray
            The solution to the problem
        '''
        return np.linalg.inv(self.A.T @ self.A + self.alpha * self.Gamma.T @ self.Gamma) @ self.A.T @ b

