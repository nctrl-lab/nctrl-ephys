import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

class PoissonRegressor:
    """
    Poisson regression using PyTorch with Tikhonov regularization.

    Parameters
    ----------
    model : str or nn.Module, optional
        The model to use. Default is 'Poisson'.
    max_iter : int, optional
        The maximum number of iterations. Default is 10000.
    tol : float, optional
        The tolerance for the optimization. Default is 1e-8.
    device : str, optional
        The device to use. Default is 'cuda' if available, otherwise 'cpu'.
    optimizer : str, optional
        The optimizer to use. Default is 'LBFGS'.
    """
    def __init__(self, model='Poisson', max_iter=10000, tol=1e-8, device=None, optimizer='LBFGS'):
        self.max_iter = max_iter
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tol = tol
        self.model = model
        self.optimizer = optimizer
        self.loss_ = None
        self.loss_full_ = None
        self.loss_null_ = None
        self.deviance_ = None
        self.deviance_null_ = None
        self.explained_deviance_ = None
        self.n_feature = None
        self.n_feature_all = None

    def fit(self, X, y, l=0.0, order=0):
        """
        Fit the Poisson regression model with Tikhonov regularization.

        Parameters
        ----------
        X : list of numpy.ndarray or numpy.ndarray
            The design matrix. If X is a list, it is concatenated along the second axis.
        y : numpy.ndarray
            The response variable.
        l : float, optional
            The regularization parameter lambda. No regularization is applied if l is 0.
        order : int, optional
            The order of the regularization. Default is 0 (ridge regularization).
            order=1 gives second-order regularization (Laplacian regularization).
            If X is a list, regularization is applied to each feature separately.

        Returns
        -------
        self : PoissonRegressor
            The fitted PoissonRegressor model.
        """
        if isinstance(X, list):
            self.n_feature = [x.shape[1] for x in X]
            X = np.concatenate(X, axis=1)
        else:
            self.n_feature = [X.shape[1]]
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        if y.ndim == 1:
            y = y.view(-1, 1)
        self.n_feature_all = X.shape[1]

        if self.model == 'Poisson':
            self.model = PoissonGLM(self.n_feature_all, y.shape[1]).to(self.device)
        elif self.model == 'Nonparametric':
            self.model = NonparametricGLM(self.n_feature_all, y.shape[1], n_bin=25).to(self.device)
        elif isinstance(self.model, nn.Module):
            self.model = self.model.to(self.device)
        else:
            raise ValueError(f"Invalid model: {self.model}")

        if self.optimizer == 'LBFGS':
            optimizer = torch.optim.LBFGS(
                self.model.parameters(),
                line_search_fn="strong_wolfe"
            )
        elif self.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters())
        else:
            raise ValueError(f"Invalid optimizer: {self.optimizer}")

        loss_fn = torch.nn.PoissonNLLLoss(log_input=False)
        D = self.tikhonov_regularizer_(order, self.n_feature).to(self.device)

        prev_loss = float('inf')
        for i in range(self.max_iter):
            if self.optimizer == 'LBFGS':
                def closure():
                    optimizer.zero_grad()
                    output = self.model(X)
                    loss = loss_fn(output, y)
                    if l > 0:
                        w = self.model.linear.weight
                        loss = loss + 0.5 * l * torch.sum((D @ w.T).pow(2))
                    loss.backward()
                    return loss
                loss = optimizer.step(closure)
            else:
                optimizer.zero_grad()
                output = self.model(X)
                loss = loss_fn(output, y)
                if l > 0:
                    w = self.model.linear.weight
                    loss = loss + 0.5 * l * torch.sum((D @ w.T).pow(2))
                loss.backward()
                optimizer.step()
            loss_val = loss.item()
            if abs(prev_loss - loss_val) < self.tol:
                print(f'Converged at {i+1} iterations')
                break
            prev_loss = loss_val

        with torch.no_grad():
            y_pred = self.model(X)
            y_mean = torch.mean(y, dim=0).expand_as(y)
            self.loss_ = loss_fn(y_pred, y).item()
            self.loss_full_ = loss_fn(y, y).item()
            self.loss_null_ = loss_fn(y_mean, y).item()
            self.deviance_ = 2 * (self.loss_ - self.loss_full_)
            self.deviance_null_ = 2 * (self.loss_null_ - self.loss_full_)
            self.explained_deviance_ = 1 - self.deviance_ / self.deviance_null_
        return self

    def transform(self, X):
        """
        Transform the design matrix X to the predicted response variable.

        Parameters
        ----------
        X : list of numpy.ndarray or numpy.ndarray
            The design matrix. If X is a list, it is concatenated along the second axis.

        Returns
        -------
        out : numpy.ndarray
            The predicted response variable.
        """
        if not isinstance(self.model, nn.Module):
            raise ValueError("Model not fit yet.")
        if isinstance(X, list):
            X = np.concatenate(X, axis=1)
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        
        self.model.eval()
        with torch.no_grad():
            out = self.model(X)
        return out.cpu().numpy()

    def tikhonov_regularizer_(self, order=0, n_feature=None):
        if isinstance(n_feature, list):
            Ls = []
            for nf in n_feature:
                if order == 0:
                    L = torch.eye(nf, device=self.device)
                else:
                    L = torch.diff(torch.eye(nf, device=self.device), n=order, dim=0) / (2 ** order)
                Ls.append(L)
            L = torch.block_diag(*Ls)
        else:
            nf = n_feature if n_feature is not None else self.n_feature_all
            if order == 0:
                L = torch.eye(nf, device=self.device)
            else:
                L = torch.diff(torch.eye(nf, device=self.device), n=order, dim=0) / (2 ** order)
        return L.t() @ L

    @property
    def coef_(self):
        if self.model is None:
            raise ValueError("Model not fit yet.")
        coef = self.model.linear.weight.detach().cpu().numpy().squeeze()
        if isinstance(self.n_feature, list) and len(self.n_feature) > 1:
            return np.split(coef, np.cumsum(self.n_feature)[:-1])
        else:
            return coef

    @property
    def intercept_(self):
        if self.model is None:
            raise ValueError("Model not fit yet.")
        return self.model.linear.bias.detach().cpu().numpy().squeeze()


class PoissonGLM(nn.Module):
    def __init__(self, n_feature, n_neuron):
        super().__init__()
        self.linear = nn.Linear(n_feature, n_neuron)

    def forward(self, x):
        return torch.exp(self.linear(x))


class NonparametricGLM(nn.Module):
    def __init__(self, n_feature, n_neuron, n_bin=25):
        super().__init__()
        self.n_bin = n_bin
        self.linear = nn.Linear(n_feature, n_neuron)
        self.norm = nn.BatchNorm1d(n_neuron)
        self.activation = PiecewiseLinear(n_bin)

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        return self.activation(x)


class PiecewiseLinear(nn.Module):
    """
    Trainable piecewise linear activation.
    """
    def __init__(self, num_points=25):
        super().__init__()
        self.num_points = num_points
        self.set()
    
    def set(self, x=None, y=None):
        if x is None:
            x = torch.linspace(-3.0, 3.0, steps=self.num_points)
        if y is None:
            y = torch.exp(x)

        slopes = torch.diff(y) / (torch.diff(x) + 1e-8)

        w = torch.zeros_like(x)
        w[0] = slopes[0]
        w[1:-1] = slopes[1:] - slopes[:-1]
        # w[-1] = -slopes[-1]

        self.register_buffer('knots', x)
        self.slopes = nn.Parameter(w)
        self.base = nn.Parameter(y[0].clone().unsqueeze(0))

    def forward(self, x):
        x_exp = x.unsqueeze(-1) - self.knots  # (..., num_points)
        relu_x = F.relu(x_exp)
        y = self.base + torch.einsum('...k,k->...', relu_x, self.slopes)
        return F.relu(y)
    
    @property
    def curve(self):
        x = torch.linspace(-4.0, 4.0, steps=self.num_points, device=self.knots.device, dtype=self.knots.dtype)
        y = self.forward(x)
        return x.detach().cpu().numpy(), y.detach().cpu().numpy()