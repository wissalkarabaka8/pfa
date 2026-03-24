import torch
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from model import BetaVAE_H, BetaVAE_B
from dataset import return_data
import os
from utils import cuda
from torch.autograd import Variable
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def reconstruction_loss(x, x_recon, distribution='gaussian'):
    batch_size = x.size(0)
    assert batch_size != 0
    if distribution == 'gaussian':
        return F.mse_loss(x_recon, x, reduction='sum') / batch_size
    else:
        return F.l1_loss(x_recon, x, reduction='sum') / batch_size


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.dim() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.dim() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)
    return total_kld, dimension_wise_kld, mean_kld


class DataGather:
    def __init__(self):
        self.data = self.get_empty_data_dict()

    def get_empty_data_dict(self):
        return dict(iter=[], recon_loss=[], total_kld=[], dim_wise_kld=[],
                    mean_kld=[], mu=[], var=[])

    def insert(self, **kwargs):
        for key in kwargs:
            self.data[key].append(kwargs[key])

    def flush(self):
        self.data = self.get_empty_data_dict()


class Solver:
    def __init__(self, args):
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.max_iter = args.max_iter
        self.global_iter = 0
        self.z_dim = args.z_dim
        self.beta = args.beta
        self.gamma = args.gamma
        self.C_max = args.C_max
        self.C_stop_iter = args.C_stop_iter
        self.objective = args.objective
        self.methode = args.methode
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.lambda_sparsity = args.lambda_sparsity
        self.nc = 3
        self.decoder_dist = 'gaussian'

        # Dual-model initialization
        self.net_H = cuda(BetaVAE_H(self.z_dim, self.nc), self.use_cuda)
        self.net_B = cuda(BetaVAE_B(self.z_dim, self.nc), self.use_cuda)
        self.optim_H = optim.Adam(self.net_H.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.optim_B = optim.Adam(self.net_B.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

        # Directories
        self.ckpt_dir = os.path.join(args.ckpt_dir, args.viz_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.output_dir = os.path.join(args.output_dir, args.viz_name)
        os.makedirs(self.output_dir, exist_ok=True)
        self.save_output = args.save_output

        # Steps
        self.gather_step = args.gather_step
        self.display_step = args.display_step
        self.save_step = args.save_step

        # Loaders
        self.data_loader = return_data(args.train_dset_dir, batch_size=args.batch_size, shuffle=True)
        self.test_loader = return_data(args.test_dset_dir, batch_size=args.batch_size, shuffle=False)

        self.gather = DataGather()

    def compute_loss(self, recon, total_kld, dim_kld):
        if self.methode == "basic":
            return recon + total_kld
        elif self.methode == "beta sparsity":
            return recon + self.beta * dim_kld.sum()
        elif self.methode == "L1 sparsity":
            return recon + self.lambda_sparsity * dim_kld.norm(p=1)
        elif self.methode == "both sparsity":
            return recon + self.beta * dim_kld.sum() + self.lambda_sparsity * dim_kld.norm(p=1)

    def net_mode(self, train=True):
        if train:
            self.net_H.train()
            self.net_B.train()
        else:
            self.net_H.eval()
            self.net_B.eval()

    def train(self):
        self.net_mode(True)
        pbar = tqdm.tqdm(total=self.max_iter)
        pbar.update(self.global_iter)
        while self.global_iter < self.max_iter:
            for x in self.data_loader:
                self.global_iter += 1
                pbar.update(1)

                x = Variable(cuda(x, self.use_cuda))

                # Forward pass
                x_recon_H, mu_H, logvar_H, _ = self.net_H(x)
                x_recon_B, mu_B, logvar_B = self.net_B(x)

                # Loss computation
                recon_H = reconstruction_loss(x, x_recon_H, self.decoder_dist)
                kld_H, dim_H, _ = kl_divergence(mu_H, logvar_H)
                loss_H = self.compute_loss(recon_H, kld_H, dim_H)

                recon_B = reconstruction_loss(x, x_recon_B, self.decoder_dist)
                kld_B, dim_B, _ = kl_divergence(mu_B, logvar_B)
                loss_B = self.compute_loss(recon_B, kld_B, dim_B)

                # Backprop
                self.optim_H.zero_grad()
                loss_H.backward()
                self.optim_H.step()

                self.optim_B.zero_grad()
                loss_B.backward()
                self.optim_B.step()

                # Gather + display
                if self.global_iter % self.gather_step == 0:
                    self.gather.insert(iter=self.global_iter, mu=mu_H.mean(0).data, var=logvar_H.exp().mean(0).data,
                                       recon_loss=recon_H.data, total_kld=kld_H.data, dim_wise_kld=dim_H.data)

                if self.global_iter % self.display_step == 0:
                    pbar.write(f"[H] iter:{self.global_iter} recon:{recon_H.item():.3f} kld:{kld_H.item():.3f}")
                    pbar.write(f"[B] iter:{self.global_iter} recon:{recon_B.item():.3f} kld:{kld_B.item():.3f}")

                # Save checkpoint
                if self.global_iter % self.save_step == 0:
                    self.save_checkpoint('last')

                if self.global_iter >= self.max_iter:
                    break
        pbar.close()
        print("[Training Finished]")

    def compute_threshold(self, train_loader=None):
        if train_loader is None:
            train_loader = self.data_loader
        scores = []
        self.net_mode(False)
        with torch.no_grad():
            for x in train_loader:
                x = Variable(cuda(x, self.use_cuda))
                x_recon_H, _, _ ,_ = self.net_H(x)
                recon_error = F.mse_loss(x_recon_H, x, reduction='none').mean(dim=1)
                scores.append(recon_error.cpu().numpy())
        scores = np.concatenate(scores)
        threshold = np.mean(scores) + 3 * np.std(scores)
        print(f"[Threshold] Computed threshold: {threshold:.4f}")
        return threshold

    def test(self, test_loader=None, threshold=None):
        if test_loader is None:
            test_loader = self.test_loader
        self.net_mode(False)
        scores = []
        with torch.no_grad():
            for x in test_loader:
                x = Variable(cuda(x, self.use_cuda))
                x_recon_H, _, _, _ = self.net_H(x)
                recon_error = F.mse_loss(x_recon_H, x, reduction='none').mean(dim=1)
                scores.append(recon_error.cpu().numpy())
        scores = np.concatenate(scores)
        if threshold is not None:
            predictions = scores > threshold
            return scores, predictions
        return scores

    def save_checkpoint(self, name):
        path_H = os.path.join(self.ckpt_dir, f"{name}_H.pt")
        path_B = os.path.join(self.ckpt_dir, f"{name}_B.pt")
        torch.save(self.net_H.state_dict(), path_H)
        torch.save(self.net_B.state_dict(), path_B)
        print(f"[Checkpoint Saved] {path_H} | {path_B}")

    def load_checkpoint(self, name):
        path_H = os.path.join(self.ckpt_dir, f"{name}_H.pt")
        path_B = os.path.join(self.ckpt_dir, f"{name}_B.pt")
        if os.path.exists(path_H) and os.path.exists(path_B):
            self.net_H.load_state_dict(torch.load(path_H, map_location='cuda' if self.use_cuda else 'cpu'))
            self.net_B.load_state_dict(torch.load(path_B, map_location='cuda' if self.use_cuda else 'cpu'))
            print(f"[Checkpoint Loaded] {path_H} | {path_B}")
    # --- LATENT SPACE ANALYSIS DURING TRAINING ---
def analyze_latent_training(self, interval=1000):
    """
    Analyse l'espace latent sur les données d'entraînement tous les 'interval' itérations.
    Calcule :
        - Variance par dimension
        - Usage factor
        - Dimensions actives
        - Corrélation (redondance)
        - Entropy
        - Sparsity
        - Efficiency score
    """
    self.net_mode(False)  # Evaluation mode
    all_mu = []
    all_logvar = []
    all_z = []

    print(f"[LEA] Running latent analysis on training data...")
    with torch.no_grad():
        for x in self.data_loader:
            x = Variable(cuda(x, self.use_cuda))
            x_recon_H, mu_H, logvar_H, _ = self.net_H(x)
            
            all_mu.append(mu_H.cpu().numpy())
            all_logvar.append(logvar_H.cpu().numpy())
            
            z = mu_H + torch.exp(logvar_H / 2) * torch.randn_like(mu_H)
            all_z.append(z.cpu().numpy())

    all_mu = np.concatenate(all_mu, axis=0)
    all_logvar = np.concatenate(all_logvar, axis=0)
    all_z = np.concatenate(all_z, axis=0)

    # 1. Variance par dimension
    dim_variance = np.var(all_mu, axis=0)
    total_variance = np.sum(dim_variance)
    usage_factor = dim_variance / (total_variance + 1e-8)

    # 2. Dimensions actives
    active_threshold = 0.01
    active_dims = usage_factor > active_threshold
    num_active_dims = np.sum(active_dims)

    # 3. Corrélation / Redondance
    correlation_matrix = np.corrcoef(all_z.T)
    np.fill_diagonal(correlation_matrix, 0)
    mean_correlation = np.mean(np.abs(correlation_matrix))

    # 4. Entropy
    entropy_per_dim = -np.mean(all_mu * np.log(np.abs(all_mu) + 1e-8), axis=0)

    # 5. Sparsity
    sparsity_per_dim = np.mean(np.abs(all_mu) < 0.1, axis=0)

    # 6. Efficiency score global
    efficiency_score = (num_active_dims / self.z_dim) * (1 - mean_correlation)

    report = {
        'dimension_variance': dim_variance,
        'usage_factor': usage_factor,
        'active_dims_mask': active_dims,
        'num_active_dims': num_active_dims,
        'total_dims': self.z_dim,
        'correlation_matrix': correlation_matrix,
        'mean_correlation': mean_correlation,
        'entropy_per_dim': entropy_per_dim,
        'sparsity_per_dim': sparsity_per_dim,
        'efficiency_score': efficiency_score,
        'total_variance': total_variance
    }

    # Optional: save histogram of variance

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['green' if active else 'red' for active in active_dims]
    ax.bar(range(len(dim_variance)), dim_variance, color=colors)
    legend_elements = [Patch(facecolor='green', label='Active'), Patch(facecolor='red', label='Inactive')]
    ax.legend(handles=legend_elements)
    ax.set_title("Variance par Dimension Latente (Training)")
    plt.tight_layout()
    hist_path = os.path.join(self.output_dir, f'latent_variance_train_iter.png')
    plt.savefig(hist_path, dpi=300)
    plt.close()
    print(f"[LEA] Latent variance histogram saved: {hist_path}")

    # Print quick summary
    print(f"[LEA] Active dims: {num_active_dims}/{self.z_dim} | Efficiency: {efficiency_score:.4f} | Mean Corr: {mean_correlation:.4f}")

    return report