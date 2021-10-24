"""
Minimal working example of a Spectral Mixture Gaussian Process.

The Spectral Mixture Gaussian Process was introduced in:
> Wilson, A., & Adams, R. (2013). Gaussian process kernels for pattern
> discovery and extrapolation. In *International conference on machine learning*
> (pp. 1067-1075). PMLR.

Notation is consistent with the paper.
"""
__date__ = "October 2021"

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import Normal, MultivariateNormal, Categorical, \
        MixtureSameFamily
from torch.utils.data import TensorDataset, DataLoader


MAX_X = 1.0 # maximum x-value
MAX_FREQ = 20.0 # maximum frequency
EPSILON = 2e-5 # for numerical stability

# Spectral content of true data-generating Gaussian Process:
TRUE_μ = torch.tensor(np.array([5, 8, 13])).to(torch.float)
ΤRUE_LOG_V = torch.tensor(np.log(np.array([0.3, 2.0, 0.8]))).to(torch.float)
TRUE_LOG_W = torch.tensor(np.log(np.array([1, 2, 1]))).to(torch.float)



class SMGP(torch.nn.Module):
    """Spectral Mixture Gaussian Process"""

    def __init__(self, q=10):
        """

        Parameters
        ----------
        q : int, optional
            Number of mixture components
        """
        super(SMGP, self).__init__()
        self.q = q
        # Initialize the GMM components on a linearly-spaced grid with equal
        # variance and equal weights.
        self.μ = torch.nn.Parameter(torch.linspace(0,MAX_FREQ,q)) # [q]
        self.log_v = torch.nn.Parameter(2.0*torch.ones(q)) # [q]
        self.log_w = torch.nn.Parameter(torch.zeros(q)) # [q]
        # Include a trainable scaling parameter.
        self.log_scale = torch.nn.Parameter(torch.zeros(1))


    def forward(self, xs, ys):
        """
        Calculate the mean data log marginal likelihood over the batch.

        Parameters
        ----------
        xs : torch.Tensor
            Shape: [t]
        ys : torch.Tensor
            Shape: [b,t]

        Returns
        -------
        log_like : torch.Tensor
            Shape: []
        """
        # Calculate the kernel matrix.
        kernel_mat = self._get_kernel_mat(xs)
        # Contrsuct the Gaussian and evaluate.
        gaussian = MultivariateNormal(torch.zeros_like(xs), kernel_mat)
        return gaussian.log_prob(ys).mean()


    def _get_kernel_mat(self, xs):
        """
        Get the kernel matrix.

        This function implements Eq. 12 in the paper with P=1.

        Parameters
        ----------
        xs : torch.Tensor
            Shape: [t]

        Returns
        -------
        kernel_mat : torch.Tensor
            Shape: [t,t]
        """
        t = xs.shape[0]
        # Get the matrix of taus.
        τ = torch.abs(xs.unsqueeze(-1) - xs.unsqueeze(-2)) # [t,t]
        τ = τ.view(1,t,t) # [1,t,t]
        # Evaluate the rest of the terms in Eq. 12.
        μ = self.μ.view(self.q,1,1) # [q,1,1]
        cos_term = torch.cos(2*np.pi*τ*μ) # [q,t,t]
        v = torch.exp(self.log_v).view(self.q,1,1) # [q,1,1]
        exp_term = torch.exp(-2*np.pi**2*τ.pow(2)*v)
        w = self.log_w.exp().view(self.q,1,1)
        kernel_mat = torch.sum(w * exp_term * cos_term, dim=0) # [t,t]
        # Scale by a learned parameter and add a small multiple of the identity
        # matrix for numerical stability.
        kernel_mat = self.log_scale.exp() * kernel_mat + EPSILON * torch.eye(t)
        return kernel_mat


    def sample(self, xs, n_samples):
        """
        Sample from the GP.

        Parameters
        ----------
        xs : torch.Tensor
        n_samples : int

        Returns
        -------
        samples : torch.Tensor
            Shape: [n,t]
        """
        with torch.no_grad():
            # Calculate the kernel matrix.
            kernel_mat = self._get_kernel_mat(xs)
            # Construct the GMM.
            mix = Categorical(logits=self.log_w)
            means = torch.zeros((self.q, xs.shape[0])) # [q,t]
            comp = MultivariateNormal(loc=means, covariance_matrix=kernel_mat)
            gmm = MixtureSameFamily(mix, comp)
            # Sample.
            samples = gmm.sample(sample_shape=(n_samples,)) # [n,t]
        return samples


    def get_psd(self, n_freqs=400):
        """
        Get the kernel power spectral density.

        Parameters
        ----------
        n_freqs : int, optional
            Number of frequencies to evaluate.

        Returns
        -------
        psd : numpy.ndarray
            Power spectral density
            Shape: [n_freqs]
        """
        freqs = torch.linspace(0, MAX_FREQ, n_freqs)
        with torch.no_grad():
            # Make the spectral GMM. We have to make two copies, one with
            # positive μ's and one with negative μ's, because spectral densities
            # are symmetric.
            mix = Categorical(logits=self.log_w)
            comp = Normal(loc=self.μ, scale=(0.5*self.log_v).exp())
            gmm = MixtureSameFamily(mix, comp)
            psd = (gmm.log_prob(freqs).exp() + gmm.log_prob(-freqs).exp()) / 2
        return psd.detach().numpy()


    def plot_samples(self, xs, ys, n_samples=2, spacing=10, ax=None):
        """
        Plot some real and generated samples.

        Parameters
        ----------
        xs : torch.Tensor
            Shape: [t]
        ys : torch.Tensor
            Shape: [b,t]
        n_samples : int, optional
        spacing : int, optional
            Vertical spacing between samples
        ax : None or matplotlib.axes._subplots.AxesSubplot
        """
        samples = self.sample(xs, n_samples)
        x = xs.detach().numpy()
        y = ys.detach().numpy()
        samples = samples.detach().numpy()
        if ax is not None:
            plt.sca(ax)
        for i in range(n_samples):
            label = 'data' if i == 0 else None
            plt.plot(x, spacing*i + y[i], c='tab:blue', label=label)
            label = 'model' if i == 0 else None
            plt.plot(x+MAX_X+0.1, spacing*i + samples[i], c='goldenrod', \
                    label=label)
        plt.axis('off')
        plt.title("Generated Samples")
        plt.legend()


    def plot_forecast(self, xs, ys, idx=0, ax=None):
        """
        Plot a GP forecast on the second half of `ys[idx]`.

        Parameters
        ----------
        xs : torch.Tensor
            Shape: [t]
        ys : torch.Tensor
            Shape: [b,t]
        idx : int, optional
            Index for `ys`.
        ax : None or matplotlib.axes._subplots.AxesSubplot
        """
        with torch.no_grad():
            t = xs.shape[0]
            temp_xs = torch.cat([xs[:t//2], xs])
            # Calculate the kernel matrix.
            kernel_mat = self._get_kernel_mat(temp_xs)
            # Calculate the forecast by Gaussian conditioning. These formulas
            # can be found here, for example, under "Conditional distributions":
            # https://en.wikipedia.org/wiki/Multivariate_normal_distribution
            k11 = kernel_mat[:t//2,:t//2]
            k12 = kernel_mat[:t//2,t//2:]
            k22 = kernel_mat[t//2:,t//2:]
            # print(k11.shape, k12.shape)
            temp = torch.linalg.solve(k11, k12).T
            # k11_inv = torch.inverse(k11)
            mean = (temp @ ys[idx,:t//2].unsqueeze(-1)).squeeze(-1)
            covar = k22 - temp @ k12
            stds = torch.diag(covar).sqrt()
        xs = xs.detach().numpy()
        ys = ys.detach().numpy()
        mean = mean.detach().numpy()
        stds = stds.detach().numpy()
        # Plot.
        if ax is not None:
            plt.sca(ax)
        plt.axvline(x=xs[t//2], c='k', ls='--', alpha=0.5)
        plt.plot(xs, mean, c='goldenrod', label="model")
        plt.fill_between(xs, mean-stds, mean+stds, fc='goldenrod', alpha=0.2)
        plt.plot(xs, ys[idx], c='tab:blue', label="data")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(loc='best')
        plt.title("Forecast")
        for dir in ['top', 'right']:
            plt.gca().spines[dir].set_visible(False)


    def plot_psd(self, true_model, n_freqs=400, ax=None):
        """
        Plot the kernel frequency representation.

        Parameters
        ----------
        true_model : SMGP
        n_freqs : int, optional
        ax : None or matplotlib.axes._subplots.AxesSubplot
        """
        model_psd = self.get_psd(n_freqs=n_freqs)
        true_psd = true_model.get_psd(n_freqs=n_freqs)
        freqs = torch.linspace(0, MAX_FREQ, n_freqs)
        # Plot.
        if ax is not None:
            plt.sca(ax)
        plt.plot(freqs, true_psd, c='tab:blue', label='ground truth')
        plt.plot(freqs, model_psd, c='goldenrod', label='model')
        plt.xlabel("Frequency")
        plt.ylabel("Spectral Density")
        plt.legend(loc='best')
        plt.title("Power Spectral Densities")
        for dir in ['top', 'right']:
            plt.gca().spines[dir].set_visible(False)



def generate_data(n=2048, t=200):
    """
    Generate synthetic data by drawing samples from a ground-truth GP.

    Returns
    -------
    ground_truth_model : SMGP
    xs : torch.Tensor
        Shape: [t]
    ys : torch.Tensor
        Shape: [b,t]
    """
    model = SMGP(q=len(TRUE_μ))
    model.μ.data = TRUE_μ
    model.log_v.data = ΤRUE_LOG_V
    model.log_w.data = TRUE_LOG_W
    xs = torch.linspace(0, MAX_X, t)
    return model, xs, model.sample(xs, n)



if __name__ == '__main__':
    # Parameters
    load_model = False
    save_model = True
    epochs = 500
    torch.manual_seed(42)

    # Get the data and make a Dataloader.
    true_model, xs, ys = generate_data() # SMGP, [t], [n,t]
    forecast_ys = true_model.sample(xs, 1) # Forecast on held-out data.
    dataset = TensorDataset(ys)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Make a model.
    model = SMGP()
    if load_model:
        checkpoint = torch.load('state.tar')
        model.load_state_dict(checkpoint)

    # Enter a training loop to optimize the kernel parameters.
    optimizer = torch.optim.Adam(model.parameters())
    for i in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            model.zero_grad()
            loss = -model(xs, *batch)
            loss.backward()
            epoch_loss += loss.item() / len(loader)
            optimizer.step()
        if i % 10 == 0:
            print(f"Epoch {i:03d}, loss: {epoch_loss:.3f}")

    # Save the model.
    if save_model:
        torch.save(model.state_dict(), 'state.tar')

    # Make some plots.
    fig, axarr = plt.subplots(nrows=3, figsize=(9,7))
    model.plot_samples(xs, ys, ax=axarr[0])
    model.plot_psd(true_model, ax=axarr[1])
    model.plot_forecast(xs, ys, ax=axarr[2])
    plt.tight_layout()
    plt.savefig('out.jpg', dpi=300)


###
