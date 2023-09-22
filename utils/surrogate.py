from botorch.models.gpytorch import GPyTorchModel
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.models import ExactGP

from utils.kernel import LaplacianKernel


class SingleTaskGP_rbf(ExactGP, GPyTorchModel):

    _num_outputs = 1

    def __init__(self, train_X, train_Y):
        super().__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            base_kernel=RBFKernel(ard_num_dims=train_X.shape[-1]),
        )
        self.to(train_X)  

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class SingleTaskGP_laplacian(ExactGP, GPyTorchModel):

    _num_outputs = 1

    def __init__(self, train_X, train_Y):
        super().__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            base_kernel=LaplacianKernel(ard_num_dims=train_X.shape[-1]),
        )
        self.to(train_X)  

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


