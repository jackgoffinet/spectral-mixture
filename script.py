"""
Spectral mixture GP

"""
__date__ = "October 2021"


import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import Normal, MultivariateNormal, Categorical, \
        MixtureSameFamily


MAX_FREQ = 20.0



class SMGP(torch.nn.Module):
    """
    Spectral Mixture Gaussian Process
    """

    def __init__(self, q=15):
        super(SMGP, self).__init__()
        self.q = q
        self.μ = torch.nn.Parameter(torch.linspace(0,MAX_FREQ,q)) # [q]
        self.log_v = torch.nn.Parameter(2.0*torch.ones(q)) # [q]
        self.log_w = torch.nn.Parameter(torch.zeros(q)) # [q]
        self.log_scale = torch.nn.Parameter(torch.zeros(1))
        self.log_noise = torch.nn.Parameter(-torch.zeros(1))


    def forward(self, xs, ys):
        """
        Calculate log marginal likelihood.

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

        Parameters
        ----------
        xs : torch.Tensor
            Shape: [t]

        Returns
        -------
        kernel_mat : torch.Tensor
            Shape: [t,t]
        """
        # Calculate the kernel matrix.
        t = xs.shape[0]
        τ = torch.abs(xs.unsqueeze(-1) - xs.unsqueeze(-2)) # [t,t]
        τ = τ.view(1,t,t) # [1,t,t]
        μ = self.μ.view(self.q,1,1) # [q,1,1]
        cos_term = torch.cos(2*np.pi*τ*μ) # [q,t,t]
        v = torch.exp(self.log_v).view(self.q,1,1) # [q,1,1]
        exp_term = torch.exp(-2*np.pi**2*τ.pow(2)*v)
        w = self.log_w.exp().view(self.q,1,1)
        kernel_mat = torch.sum(w * exp_term * cos_term, dim=0) # [t,t]
        noise = self.log_noise.exp()
        kernel_mat = self.log_scale.exp() * kernel_mat + noise * torch.eye(t)
        return kernel_mat


    def plot_sample(self, xs, ys, n_samples=3, spacing=6):
        """Plot some real and generated samples."""
        with torch.no_grad():
            t = xs.shape[0]
            # Calculate the kernel matrix.
            kernel_mat = self._get_kernel_mat(xs)
            # Construct the GMM.
            mix = Categorical(logits=self.log_w)
            means = torch.zeros((self.q,t)) # [t]
            comp = MultivariateNormal(loc=means, covariance_matrix=kernel_mat)
            gmm = MixtureSameFamily(mix, comp)
            # Sample.
            samples = gmm.sample(sample_shape=(n_samples,))
        x = xs.detach().cpu().numpy()
        y = ys.detach().cpu().numpy()
        samples = samples.detach().cpu().numpy()
        for i in range(n_samples):
            label = 'generated' if i == 0 else None
            plt.plot(x, spacing*i + samples[i], c='darkorchid', label=label)
            label = 'real' if i == 0 else None
            plt.plot(x, spacing*i + y[i], c='goldenrod', label=label)
        plt.axis('off')
        plt.legend()
        plt.savefig('samples.pdf')
        plt.close('all')


    def plot_forecast(self, xs, ys, idx=2):
        """Plot the GP forecast."""
        with torch.no_grad():
            t = xs.shape[0]
            # Calculate the kernel matrix.
            kernel_mat = self._get_kernel_mat(xs)
            k11 = kernel_mat[:t//2,:t//2]
            k12 = kernel_mat[:t//2,t//2:]
            k22 = kernel_mat[t//2:,t//2:]
            k11_inv = torch.inverse(k11)
            mean = k12.T @ k11_inv @ ys[idx,:t//2].unsqueeze(-1)
            mean = mean.squeeze(-1)
            covar = k22 - k12.T @ k11_inv @ k12
            stds = torch.diag(covar).sqrt()
        xs = xs.detach().cpu().numpy()
        ys = ys.detach().cpu().numpy()
        mean = mean.detach().cpu().numpy()
        stds = stds.detach().cpu().numpy()
        plt.plot(xs, ys[idx])
        plt.plot(xs[t//2:], mean, c='goldenrod')
        plt.fill_between(xs[t//2:], mean-stds, mean+stds, fc='goldenrod', alpha=0.1)
        plt.savefig('forecast.pdf')
        plt.close('all')


    def plot_psd(self):
        """Plot the kernel frequency representation."""
        with torch.no_grad():
            # Make the spectral GMM.
            mix = Categorical(logits=self.log_w)
            comp = Normal(loc=self.μ, scale=(0.5*self.log_v).exp())
            gmm = MixtureSameFamily(mix, comp)
            xs = torch.linspace(0,MAX_FREQ,200)
            probs_1 = gmm.log_prob(xs).exp()
            comp = Normal(loc=-self.μ, scale=(0.5*self.log_v).exp())
            gmm = MixtureSameFamily(mix, comp)
            xs = torch.linspace(0,MAX_FREQ,200)
            probs_2 = gmm.log_prob(xs).exp()
            probs = probs_1 + probs_2
        probs = probs.detach().cpu().numpy()
        xs = xs.detach().cpu().numpy()
        plt.plot(xs, probs)
        plt.xlabel("Frequency")
        plt.ylabel("Spectral Density")
        plt.savefig('kernel_psd.pdf')
        plt.close('all')



if __name__ == '__main__':
    load_model = False
    
    # Load the data.
    xs = np.linspace(0,1,200)
    ys = np.zeros((1024, 200))
    freqs = 2*np.pi*np.array([5, 11])
    for i in range(len(ys)):
        temp_freqs = freqs + 0.1 * np.random.randn(1)
        phis = 2 * np.pi * np.random.rand(len(freqs))
        ys[i] = sum(np.sin(freq * xs + phi) for phi, freq in zip(phis,temp_freqs))

    xs = torch.tensor(xs).to(torch.float)
    ys = torch.tensor(ys).to(torch.float)

    # Make a model.
    model = SMGP()

    if load_model:
        checkpoint = torch.load('state.tar')
        model.load_state_dict(checkpoint)

    model.plot_forecast(xs, ys)

    # Enter a training loop.
    optimizer = torch.optim.Adam(model.parameters())
    for i in range(5000):
        model.zero_grad()
        batch = ys[np.random.permutation(ys.shape[0])[:128]]
        loss = -model(xs, batch)
        loss.backward()
        if i % 10 == 0:
            print(f"Epoch {i}, loss: {loss.item()}")
        optimizer.step()

    # Plot some things.
    model.plot_psd()
    model.plot_sample(xs, ys)
    model.plot_forecast(xs, ys)

    # Save the model.
    torch.save(model.state_dict(), 'state.tar')


###
