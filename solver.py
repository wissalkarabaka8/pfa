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
    total_kld = klds.sum(1).mean(0, True)   # vraie KL globale (somme sur toutes les dim)
    dimension_wise_kld = klds.mean(0)        # KL moyenne par batch pour chaque dim
    mean_kld = klds.mean(1).mean(0, True)    # KL moyenne par dim
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

        # Threshold (calculé après training)
        self.threshold = None

    # ─────────────────────────────────────────────────────────────────────
    # LOSS
    # ─────────────────────────────────────────────────────────────────────
    def compute_loss(self, recon, total_kld, dim_kld):
        """
        - 'basic':           recon + KL
        - 'beta sparsity':   recon + beta * total_kld
        - 'L1 sparsity':     recon + lambda * L1(dim_kld)
        - 'both sparsity':   recon + beta * total_kld + lambda * L1(dim_kld)
        """
        if self.methode == "basic":
            return recon + total_kld
        elif self.methode == "beta sparsity":
            return recon + self.beta * total_kld
        elif self.methode == "L1 sparsity":
            return recon + self.lambda_sparsity * dim_kld.norm(p=1)
        elif self.methode == "both sparsity":
            return recon + self.beta * total_kld + self.lambda_sparsity * dim_kld.norm(p=1)
        else:
            raise ValueError(f"Unknown methode: '{self.methode}'")

    # ─────────────────────────────────────────────────────────────────────
    # NET MODE
    # ─────────────────────────────────────────────────────────────────────
    def net_mode(self, train=True):
        if train:
            self.net_H.train()
            self.net_B.train()
        else:
            self.net_H.eval()
            self.net_B.eval()

    # ─────────────────────────────────────────────────────────────────────
    # TRAIN
    # ─────────────────────────────────────────────────────────────────────
    def train(self):
        self.net_mode(True)
        pbar = tqdm.tqdm(total=self.max_iter)
        pbar.update(self.global_iter)

        while self.global_iter < self.max_iter:
            for x in self.data_loader:
                self.global_iter += 1
                pbar.update(1)

                x = cuda(x, self.use_cuda)

                # Forward pass
                x_recon_H, mu_H, logvar_H, _ = self.net_H(x)
                x_recon_B, mu_B, logvar_B, _ = self.net_B(x)

                # Loss
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

                # Gather
                if self.global_iter % self.gather_step == 0:
                    self.gather.insert(
                        iter=self.global_iter,
                        mu=mu_H.mean(0).data,
                        var=logvar_H.exp().mean(0).data,
                        recon_loss=recon_H.data,
                        total_kld=kld_H.data,
                        dim_wise_kld=dim_H.data,
                    )

                if self.global_iter % self.display_step == 0:
                    pbar.write(f"[H] iter:{self.global_iter} recon:{recon_H.item():.3f} kld:{kld_H.item():.3f}")
                    pbar.write(f"[B] iter:{self.global_iter} recon:{recon_B.item():.3f} kld:{kld_B.item():.3f}")

                if self.global_iter % self.save_step == 0:
                    self.save_checkpoint('last')

                if self.global_iter >= self.max_iter:
                    break

        pbar.close()
        print("[Training Finished]")

        # ✅ Calculer le threshold automatiquement après le training
        print("[INFO] Computing threshold on training data...")
        self.threshold = self.compute_threshold()
        print(f"[INFO] Threshold set to: {self.threshold:.4f}")

    # ─────────────────────────────────────────────────────────────────────
    # RECONSTRUCTION SCORES
    # ─────────────────────────────────────────────────────────────────────
    def _recon_scores(self, net, loader):
        """Calcule l'erreur de reconstruction par image pour tout un loader."""
        scores = []
        with torch.no_grad():
            for x in loader:
                x = cuda(x, self.use_cuda)
                x_recon, *_ = net(x)
                recon_dims = list(range(1, x.dim()))
                recon_error = F.mse_loss(x_recon, x, reduction='none').mean(dim=recon_dims)
                scores.append(recon_error.cpu().numpy())
        return np.concatenate(scores)

    # ─────────────────────────────────────────────────────────────────────
    # THRESHOLD
    # ─────────────────────────────────────────────────────────────────────
    def compute_threshold(self, train_loader=None):
        """Calcule le seuil via la règle des 3-sigma sur les données d'entraînement."""
        if train_loader is None:
            train_loader = self.data_loader
        self.net_mode(False)
        scores = self._recon_scores(self.net_H, train_loader)
        threshold = np.mean(scores) + 3 * np.std(scores)
        print(f"[Threshold] Computed threshold: {threshold:.4f}")
        return threshold

    def save_threshold(self):
        """✅ Sauvegarde le threshold dans un fichier .npy"""
        if self.threshold is None:
            print("[WARNING] Threshold est None — calcul avant sauvegarde...")
            self.threshold = self.compute_threshold()
        path = os.path.join(self.ckpt_dir, 'threshold.npy')
        np.save(path, np.array(self.threshold))
        print(f"[Threshold Saved] {path}  (valeur: {self.threshold:.4f})")

    def load_threshold(self):
        """✅ Charge le threshold depuis le fichier .npy"""
        path = os.path.join(self.ckpt_dir, 'threshold.npy')
        if os.path.exists(path):
            self.threshold = float(np.load(path))
            print(f"[Threshold Loaded] {self.threshold:.4f}")
        else:
            print("[WARNING] Fichier threshold introuvable → recalcul...")
            self.threshold = self.compute_threshold()

    # ─────────────────────────────────────────────────────────────────────
    # TEST  (dossier entier → REAL / FAKE)
    # ─────────────────────────────────────────────────────────────────────
    def test(self, test_loader=None, threshold=None):
        """
        Évalue les images de test et retourne REAL ou FAKE pour chaque image.

        Returns
        -------
        results : list of dict
            Chaque dict contient :
                - 'image_idx' : int
                - 'score_H'   : float
                - 'score_B'   : float
                - 'label_H'   : 'REAL' ou 'FAKE'
                - 'label_B'   : 'REAL' ou 'FAKE'
        """
        if test_loader is None:
            test_loader = self.test_loader

        if threshold is None:
            if self.threshold is None:
                print("[WARNING] Threshold non calculé — lancement de compute_threshold()...")
                self.threshold = self.compute_threshold()
            threshold = self.threshold

        self.net_mode(False)

        scores_H = self._recon_scores(self.net_H, test_loader)
        scores_B = self._recon_scores(self.net_B, test_loader)

        results = []
        print(f"\n{'='*55}")
        print(f"  {'#':<5} {'Score_H':>10} {'Label_H':>10} {'Score_B':>10} {'Label_B':>10}")
        print(f"{'='*55}")

        for i, (sh, sb) in enumerate(zip(scores_H, scores_B)):
            label_H = "FAKE" if sh > threshold else "REAL"
            label_B = "FAKE" if sb > threshold else "REAL"
            results.append({
                'image_idx': i,
                'score_H':   float(sh),
                'score_B':   float(sb),
                'label_H':   label_H,
                'label_B':   label_B,
            })
            print(f"  {i:<5} {sh:>10.4f} {label_H:>10} {sb:>10.4f} {label_B:>10}")

        print(f"{'='*55}")
        print(f"[INFO] Threshold utilisé : {threshold:.4f}")
        print(f"[INFO] FAKE détectés (H) : {sum(1 for r in results if r['label_H'] == 'FAKE')}/{len(results)}")
        print(f"[INFO] FAKE détectés (B) : {sum(1 for r in results if r['label_B'] == 'FAKE')}/{len(results)}")

        return results

    # ─────────────────────────────────────────────────────────────────────
    # PREDICT  (une seule image → REAL / FAKE)
    # ─────────────────────────────────────────────────────────────────────
    def predict_single(self, image_tensor, threshold=None):
        """
        Prédit si UNE SEULE image est REAL ou FAKE.

        Parameters
        ----------
        image_tensor : torch.Tensor de shape (1, 3, 64, 64)
        threshold    : float (utilise self.threshold si non fourni)

        Returns
        -------
        dict avec score_H, score_B, label_H, label_B
        """
        if threshold is None:
            if self.threshold is None:
                print("[WARNING] Threshold non calculé — lancement de compute_threshold()...")
                self.threshold = self.compute_threshold()
            threshold = self.threshold

        self.net_mode(False)
        with torch.no_grad():
            image_tensor = cuda(image_tensor, self.use_cuda)
            recon_dims = list(range(1, image_tensor.dim()))

            x_recon_H, *_ = self.net_H(image_tensor)
            score_H = F.mse_loss(x_recon_H, image_tensor, reduction='none').mean(dim=recon_dims).item()

            x_recon_B, *_ = self.net_B(image_tensor)
            score_B = F.mse_loss(x_recon_B, image_tensor, reduction='none').mean(dim=recon_dims).item()

        label_H = "FAKE" if score_H > threshold else "REAL"
        label_B = "FAKE" if score_B > threshold else "REAL"

        print(f"\n{'='*40}")
        print(f"  Threshold  : {threshold:.4f}")
        print(f"  Score H    : {score_H:.4f}  →  {label_H}")
        print(f"  Score B    : {score_B:.4f}  →  {label_B}")
        print(f"{'='*40}")

        return {
            'score_H': score_H,
            'score_B': score_B,
            'label_H': label_H,
            'label_B': label_B,
        }

    # ─────────────────────────────────────────────────────────────────────
    # CHECKPOINT
    # ─────────────────────────────────────────────────────────────────────
    def save_checkpoint(self, name):
        path_H = os.path.join(self.ckpt_dir, f"{name}_H.pt")
        path_B = os.path.join(self.ckpt_dir, f"{name}_B.pt")
        torch.save(self.net_H.state_dict(), path_H)
        torch.save(self.net_B.state_dict(), path_B)
        print(f"[Checkpoint Saved] {path_H} | {path_B}")

    def load_checkpoint(self, name):
        """✅ Charge les poids — NE recalcule PAS le threshold (utiliser load_threshold())"""
        path_H = os.path.join(self.ckpt_dir, f"{name}_H.pt")
        path_B = os.path.join(self.ckpt_dir, f"{name}_B.pt")
        if os.path.exists(path_H) and os.path.exists(path_B):
            self.net_H.load_state_dict(
                torch.load(path_H, map_location='cuda' if self.use_cuda else 'cpu')
            )
            self.net_B.load_state_dict(
                torch.load(path_B, map_location='cuda' if self.use_cuda else 'cpu')
            )
            print(f"[Checkpoint Loaded] {path_H} | {path_B}")
        else:
            raise FileNotFoundError(f"❌ Checkpoint introuvable : {path_H} ou {path_B}")

    # ─────────────────────────────────────────────────────────────────────
    # ANALYSE ESPACE LATENT
    # ─────────────────────────────────────────────────────────────────────
    def analyze_latent_training(self, interval=1000):
        """
        Analyse the latent space on training data.
        Metrics: Variance, Active dims, Correlation, Entropy, Sparsity, Efficiency.
        """
        self.net_mode(False)

        all_mu, all_logvar, all_z = [], [], []
        batch_active_dims = []

        print("[LEA] Running latent analysis on training data...")

        with torch.no_grad():
            for x in self.data_loader:
                x = cuda(x, self.use_cuda)
                _, mu_H, logvar_H, _ = self.net_H(x)

                all_mu.append(mu_H.cpu().numpy())
                all_logvar.append(logvar_H.cpu().numpy())

                z = mu_H + torch.exp(logvar_H / 2) * torch.randn_like(mu_H)
                all_z.append(z.cpu().numpy())

                var_batch = torch.var(mu_H, dim=0, unbiased=False)
                batch_active_dims.append((var_batch > 0.01).float().sum().item())

        all_mu     = np.concatenate(all_mu,     axis=0)
        all_logvar = np.concatenate(all_logvar, axis=0)
        all_z      = np.concatenate(all_z,      axis=0)

        def _save(fname):
            p = os.path.join(self.output_dir, fname)
            plt.tight_layout()
            plt.savefig(p, dpi=300)
            plt.close()
            print(f"[LEA] Saved: {p}")

        # 1. Variance
        dim_variance   = np.var(all_mu, axis=0)
        total_variance = float(np.sum(dim_variance))

        # 2. Active dimensions
        active_dims     = dim_variance > 0.01
        num_active_dims = int(np.sum(active_dims))
        active_ratio    = num_active_dims / self.z_dim

        # 3. Correlation
        corr = np.corrcoef(all_z.T)
        corr = np.nan_to_num(corr)
        np.fill_diagonal(corr, 0)
        mean_correlation = float(np.mean(np.abs(corr)))

        # 4. Entropie gaussienne correcte (depuis logvar)
        entropy_per_dim = 0.5 * np.mean(1 + all_logvar, axis=0)

        # 5. Sparsity
        sparsity_per_dim = np.mean(np.abs(all_mu) < 0.1, axis=0)

        # 6. Efficiency
        efficiency_score = active_ratio * (1 - mean_correlation)

        # Print
        print(f"\n[LEA] Active dims   : {num_active_dims}/{self.z_dim}")
        print(f"[LEA] Active ratio  : {active_ratio:.4f}")
        print(f"[LEA] Efficiency    : {efficiency_score:.4f}")
        print(f"[LEA] Mean corr     : {mean_correlation:.4f}")
        print(f"[LEA] Total variance: {total_variance:.4f}")

        if active_ratio > 0.9:
            print("→ No sparsity — all dimensions are used")
        elif active_ratio > 0.3:
            print("→ Moderate sparsity")
        else:
            print("→ Risk of posterior collapse ⚠️")

        # Plots
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(all_z.flatten(), bins=100)
        ax.set_title("Distribution of Latent z")
        ax.set_xlabel("z values"); ax.set_ylabel("Frequency"); ax.grid(True)
        _save('latent_z_distribution.png')

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(batch_active_dims, bins=20)
        ax.set_title("Active Dimensions per Batch")
        ax.set_xlabel("Number of Active Dimensions"); ax.set_ylabel("Frequency"); ax.grid(True)
        _save('latent_batch_active_dims.png')

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(dim_variance, marker='o')
        ax.set_title("Variance per Latent Dimension")
        ax.set_xlabel("Latent Dimension"); ax.set_ylabel("Variance"); ax.grid(True)
        _save('latent_variance_curve.png')

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['green' if a else 'red' for a in active_dims]
        ax.bar(range(len(dim_variance)), dim_variance, color=colors)
        ax.legend(handles=[
            Patch(facecolor='green', label='Active'),
            Patch(facecolor='red',   label='Inactive'),
        ])
        ax.set_title("Variance per Latent Dimension (Active / Inactive)")
        _save('latent_variance_bar.png')

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(len(entropy_per_dim)), entropy_per_dim)
        ax.set_title("Entropy per Dimension (Gaussian)"); ax.grid(True)
        _save('latent_entropy.png')

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(len(sparsity_per_dim)), sparsity_per_dim)
        ax.set_title("Sparsity per Dimension"); ax.grid(True)
        _save('latent_sparsity.png')

        fig, ax = plt.subplots(figsize=(10, 5))
        dimension_counts = np.sum(np.abs(all_z) > 0.01, axis=0)
        ax.bar(range(len(dimension_counts)), dimension_counts, color='blue')
        ax.set_title("Frequency of Active Values per Latent Dimension")
        ax.set_xlabel("Latent Dimension"); ax.set_ylabel("Count over dataset"); ax.grid(True)
        _save('latent_dimension_frequency.png')

        return {
            'dimension_variance': dim_variance,
            'active_dims_mask':   active_dims,
            'num_active_dims':    num_active_dims,
            'active_ratio':       active_ratio,
            'correlation_matrix': corr,
            'mean_correlation':   mean_correlation,
            'entropy_per_dim':    entropy_per_dim,
            'sparsity_per_dim':   sparsity_per_dim,
            'efficiency_score':   efficiency_score,
            'total_variance':     total_variance,
            'batch_active_dims':  batch_active_dims,
        }

    # ─────────────────────────────────────────────────────────────────────
    # COMPARE
    # ─────────────────────────────────────────────────────────────────────
    def compare_models(self):
        """Swap net_H/net_B avec try/finally pour garantir la restauration."""
        print("\n========== MODEL COMPARISON ==========\n")

        print(">>> Analyzing Model H")
        results_H = self.analyze_latent_training()

        print("\n>>> Analyzing Model B")
        original_net_H = self.net_H
        try:
            self.net_H = self.net_B
            results_B = self.analyze_latent_training()
        finally:
            self.net_H = original_net_H  # toujours restauré même si exception

        return {"H": results_H, "B": results_B}

    def print_comparison(self, results):
        print("\n===== FINAL COMPARISON =====\n")
        for model in ["H", "B"]:
            r = results[model]
            print(f"Model {model}:")
            print(f"  Active dims     : {r['num_active_dims']}")
            print(f"  Active ratio    : {r['active_ratio']:.4f}")
            print(f"  Efficiency      : {r['efficiency_score']:.4f}")
            print(f"  Mean correlation: {r['mean_correlation']:.4f}\n")

    def select_best_model(self, results):
        best = max(results, key=lambda m: results[m]['efficiency_score'])
        print(f"\n🏆 BEST MODEL: {best}")
        return best