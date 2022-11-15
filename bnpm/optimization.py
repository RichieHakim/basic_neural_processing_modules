import torch


class Convergence_checker:
    """
    Checks for convergence during an optimization process. Uses Ordinary Least Squares (OLS) to 
     fit a line to the last 'window_convergence' number of iterations.

    RH 2022
    """
    def __init__(
        self,
        tol_convergence=1e-2,
        window_convergence=100,
    ):
        """
        Initialize the convergence checker.
        
        Args:
            tol_convergence (float): 
                Tolerance for convergence.
                Corresponds to the slope of the line that is fit.
            window_convergence (int):
                Number of iterations to use for fitting the line.
        """
        self.window_convergence = window_convergence
        self.tol_convergence = tol_convergence

        self.line_regressor = torch.cat((torch.linspace(0,1,window_convergence)[:,None], torch.ones((window_convergence,1))), dim=1)

    def OLS(self, y):
        """
        Ordinary least squares.
        Fits a line and bias term (stored in self.line_regressor)
         to y input.
        """
        X = self.line_regressor
        theta = torch.inverse(X.T @ X) @ X.T @ y
        y_rec = X @ theta
        bias = theta[-1]
        theta = theta[:-1]

        return theta, y_rec, bias

    def __call__(
        self,
        loss_history,
    ):
        """
        Forward pass of the convergence checker.
        Checks if the last 'window_convergence' number of iterations are
         within 'tol_convergence' of the line fit.

        Args:
            loss_history (list or array):
                List of loss values for the last 'window_convergence' 
                 number of iterations.

        Returns:
            diff_window_convergence (float):
                Difference between first and last element of the fit line
                 over the range of 'window_convergence'.
            loss_smooth (float):
                The mean loss over 'window_convergence'.
            converged (bool):
                True if the 'diff_window_convergence' is less than
                 'tol_convergence'.
        """
        if len(loss_history) < self.window_convergence:
            return torch.nan, torch.nan, False
        loss_window = torch.as_tensor(loss_history[-self.window_convergence:], device='cpu', dtype=torch.float32)
        theta, y_rec, bias = self.OLS(y=loss_window)

        diff_window_convergence = (y_rec[-1] - y_rec[0])
        loss_smooth = loss_window.mean()
        converged = True if torch.abs(diff_window_convergence) < self.tol_convergence else False
        return diff_window_convergence.item(), loss_smooth.item(), converged