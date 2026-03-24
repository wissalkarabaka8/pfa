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


def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0
    if distribution == 'gaussian':
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    else:
        recon_loss = None

    return recon_loss


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


class DataGather(object):
    def __init__(self):
        self.data = self.get_empty_data_dict()

    def get_empty_data_dict(self):
        return dict(iter=[],
                    recon_loss=[],
                    total_kld=[],
                    dim_wise_kld=[],
                    mean_kld=[],
                    mu=[],
                    var=[])

    def insert(self, **kwargs):
        for key in kwargs:
            self.data[key].append(kwargs[key])

    def flush(self):
        self.data = self.get_empty_data_dict()


class Solver(object):
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
        self.model = args.model
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.lambda_sparsity=args.lambda_sparsity
        
        # Nombre de canaux (RGB)
        self.nc = 3
        # Distribution pour la perte de reconstruction
        self.decoder_dist = 'gaussian'

        if args.model == 'H':
            net = BetaVAE_H
        elif args.model == 'B':
            net = BetaVAE_B
        else:
            raise NotImplementedError('only support model H or B')

        self.net = cuda(net(self.z_dim, self.nc), self.use_cuda)
        self.optim = optim.Adam(self.net.parameters(), lr=self.lr,
                                    betas=(self.beta1, self.beta2))
        
        
        self.ckpt_dir = os.path.join(args.ckpt_dir, args.viz_name)#crée un dossier pour les checkpoints(contient des snapshots du modèle pendant l'entraînement)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt_name = args.ckpt_name
        if self.ckpt_name is not None:
            self.load_checkpoint(self.ckpt_name)

        self.save_output = args.save_output
        self.output_dir = os.path.join(args.output_dir, args.viz_name)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Flag de visualisation
        self.viz_on = True

        self.gather_step = args.gather_step
        self.display_step = args.display_step
        self.save_step = args.save_step
        
        # DataLoader pour l'entraînement (images réelles/normales)
        self.data_loader = return_data(args.train_dset_dir, batch_size=args.batch_size, shuffle=True)
        
        # DataLoader pour le test (mélange normal + anomalies)
        self.test_loader = return_data(args.test_dset_dir, batch_size=args.batch_size, shuffle=False)

        self.gather = DataGather()

    def train(self):
        self.net_mode(train=True)#met le modèle en mode entraînement 
        C_max = Variable(cuda(torch.FloatTensor([self.C_max]), self.use_cuda))#convertit C_max en un tenseur de type float et le place sur le GPU si use_cuda est vrai
        out = False

        pbar = tqdm(total=self.max_iter)#crée une barre de progression pour suivre l'avancement de l'entraînement
        pbar.update(self.global_iter)#met à jour la barre de progression avec le nombre d'itérations déjà effectuées (utile si on reprend l'entraînement à partir d'un checkpoint)
        while not out:
            for x in self.data_loader:#chaeger les batches du DataLoader
                self.global_iter += 1
                pbar.update(1)

                x = Variable(cuda(x, self.use_cuda))#place le batch d'images sur le GPU si use_cuda est vrai et le convertit en un objet Variable (nécessaire pour le calcul du gradient)
                x_recon, mu, logvar = self.net(x)#passe le batch d'images à travers le modèle 
                recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)#
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
                
                if self.methode=="beta sparsity":
                    beta_vae_loss = recon_loss + self.beta*dim_wise_kld.sum()
                elif self.methode=="L1 sparsity":
                    beta_vae_loss = recon_loss + self.lambda_sparsity*(dim_wise_kld.norm(p=1))
                elif self.methode=="both sparsity":
                    beta_vae_loss = recon_loss + self.beta*dim_wise_kld.sum() + self.lambda_sparsity*(dim_wise_kld.norm(p=1))

                self.optim.zero_grad()# Réinitialise les gradients
                beta_vae_loss.backward()# Rétropropagation
                self.optim.step()# Mise à jour des poids

                if self.viz_on and self.global_iter%self.gather_step == 0:#collecte les données pour le suivi de l'entraînement et les affiche si viz_on est vrai et que le nombre d'itérations est un multiple de gather_step
                    self.gather.insert(iter=self.global_iter,
                                    mu=mu.mean(0).data, var=logvar.exp().mean(0).data,
                                    recon_loss=recon_loss.data, total_kld=total_kld.data,
                                    dim_wise_kld=dim_wise_kld.data, mean_kld=mean_kld.data)

                if self.global_iter%self.display_step == 0:#affiche les résultats de l'entraînement sur la console si le nombre d'itérations est un multiple de display_step
                    pbar.write('[{}] recon_loss:{:.3f} total_kld:{:.3f} mean_kld:{:.3f}'.format(
                        self.global_iter, recon_loss.item(), total_kld.item(), mean_kld.item()))

                    var = logvar.exp().mean(0).data
                    var_str = ''
                    for j, var_j in enumerate(var):
                        var_str += 'var{}:{:.4f} '.format(j+1, var_j)
                    pbar.write(var_str)

                    if self.objective == 'B':
                        pbar.write('C:{:.3f}'.format(C_max.item()))

                    if self.viz_on or self.save_output:
                        self.viz_traverse()

                if self.global_iter%self.save_step == 0:#sauvegarde le modèle (checkpoint) si le nombre d'itérations est un multiple de save_step
                    self.save_checkpoint('last')
                    pbar.write('Saved checkpoint(iter:{})'.format(self.global_iter))

                if self.global_iter%50000 == 0:#sauvegarde le modèle (checkpoint) tous les 50000 itérations
                    self.save_checkpoint(str(self.global_iter))

                if self.global_iter >= self.max_iter:#arrête l'entraînement si le nombre d'itérations atteint max_iter
                    out = True
                    break

        pbar.write("[Training Finished]")
        pbar.close()

    def compute_anomaly_score(self, x):
        """
        Calcule le score d'anomalie (reconstruction error) pour les données
        
        Args:
            x: tenseur d'images
            
        Returns:
            numpy array des scores de reconstruction error
        """
        self.net_mode(train=False)
        with torch.no_grad():
            x = Variable(cuda(x, self.use_cuda))
            x_recon, _, _ = self.net(x)
            recon_error = F.mse_loss(x_recon, x, reduction='none').mean(dim=1)
        return recon_error.detach().cpu().numpy()

    def test(self, test_loader, threshold=None):
        """
        Teste le modèle VAE et calcule les scores d'anomalie
        
        Args:
            test_loader: DataLoader avec les données de test
            threshold: seuil d'anomalie (optionnel)
            
        Returns:
            anomaly_scores: array des scores d'anomalie
        """
        self.net_mode(train=False)
        anomaly_scores = []
        
        with torch.no_grad():
            for x in test_loader:
                x = Variable(cuda(x, self.use_cuda))
                x_recon, _, _ = self.net(x)
                recon_error = F.mse_loss(x_recon, x, reduction='none').mean(dim=1)
                anomaly_scores.append(recon_error.cpu().numpy())
        
        anomaly_scores = np.concatenate(anomaly_scores)
        return anomaly_scores

    def detect_anomalies(self, anomaly_scores, threshold):
        """
        Détecte les anomalies basées sur un seuil
        
        Args:
            anomaly_scores: array des scores d'anomalie
            threshold: seuil de détection d'anomalie
            
        Returns:
            predictions: array booléen (True = anomalie, False = normal)
        """
        predictions = anomaly_scores > threshold
        num_anomalies = predictions.sum()
        print(f"[Anomaly Detection] Anomalies détectées: {num_anomalies}/{len(predictions)}")
        return predictions

    def net_mode(self, train=True):
        """Met le modèle en mode train ou eval"""
        if train:
            self.net.train()
        else:
            self.net.eval()

    def save_checkpoint(self, name):
        """Sauvegarde le checkpoint du modèle"""
        torch.save(self.net.state_dict(), 
                   os.path.join(self.ckpt_dir, f'{name}.pt'))

    def load_checkpoint(self, name):
        """Charge un checkpoint du modèle"""
        ckpt_path = os.path.join(self.ckpt_dir, f'{name}.pt')
        if os.path.exists(ckpt_path):
            self.net.load_state_dict(torch.load(ckpt_path, map_location='cpu' if not self.use_cuda else None))
            print(f'Loaded checkpoint: {name}')

    def viz_traverse(self):
        """Stub pour visualisation de traversée latente"""
        pass

    def analyze_latent_efficiency(self, test_loader=None):
        """
        Analyse l'efficacité de l'utilisation de l'espace latent sur les données de test
        
        Mesures :
        - Variance par dimension latente
        - Usage factor (contribution de chaque dimension)
        - Redondance (corrélations entre dimensions)
        - Score d'efficacité global
        
        Returns:
            rapport_lea: dictionnaire avec toutes les métriques d'efficacité
        """
        if test_loader is None:
            test_loader = self.test_loader
        
        self.net_mode(train=False)
        
        all_mu = []
        all_logvar = []
        all_z = []
        
        print("[LEA] Traitement des données de test...")
        with torch.no_grad():
            for x in test_loader:
                x = Variable(cuda(x, self.use_cuda))
                # Extraction de l'encoding
                if isinstance(self.net, (BetaVAE_H, BetaVAE_B)):
                    # Assume le modèle retourne (x_recon, mu, logvar)
                    _, mu, logvar = self.net(x)
                else:
                    # Fallback
                    x_recon, mu, logvar = self.net(x)
                
                all_mu.append(mu.cpu().numpy())
                all_logvar.append(logvar.cpu().numpy())
                # Sample z
                z = mu + torch.exp(logvar / 2) * torch.randn_like(mu)
                all_z.append(z.cpu().numpy())
        
        # Concatener les batches
        all_mu = np.concatenate(all_mu, axis=0)
        all_logvar = np.concatenate(all_logvar, axis=0)
        all_z = np.concatenate(all_z, axis=0)
        
        # 1. VARIANCE PAR DIMENSION
        dim_variance = np.var(all_mu, axis=0)
        dim_mean = np.mean(all_mu, axis=0)
        
        # 2. USAGE FACTOR - Combien chaque dimension contribue à la variance totale
        total_variance = np.sum(dim_variance)
        usage_factor = dim_variance / (total_variance + 1e-8)
        
        # 3. DIMENSIONS ACTIVES - Threshold pour déterminer si une dimension est utilisée
        active_threshold = 0.01  # Les dimensions avec <1% de la variance totale sont inactives
        active_dims = usage_factor > active_threshold
        num_active_dims = np.sum(active_dims)
        
        # 4. REDONDANCE - Corrélation entre dimensions
        correlation_matrix = np.corrcoef(all_z.T)
        # Ignorer la diagonal
        np.fill_diagonal(correlation_matrix, 0)
        # Moyenne de corrélation absolue
        mean_correlation = np.mean(np.abs(correlation_matrix))
        
        # 5. ENTROPY DE CHAQUE DIMENSION (mesure de l'utilisation)
        entropy_per_dim = -np.mean(all_mu * np.log(np.abs(all_mu) + 1e-8), axis=0)
        
        # 6. SPARSITY - Pourcentage de valeurs proches de 0
        sparsity_per_dim = np.mean(np.abs(all_mu) < 0.1, axis=0)
        
        # 7. SCORE D'EFFICACITÉ GLOBAL
        # Idéal : toutes les dimensions fortement utilisées et non corrélées
        efficiency_score = (num_active_dims / self.z_dim) * (1 - mean_correlation)
        
        rapport_lea = {
            'dimension_variance': dim_variance,
            'dimension_mean': dim_mean,
            'usage_factor': usage_factor,
            'active_dimensions': num_active_dims,
            'total_dimensions': self.z_dim,
            'active_dims_mask': active_dims,
            'correlation_matrix': correlation_matrix,
            'mean_correlation': mean_correlation,
            'entropy_per_dim': entropy_per_dim,
            'sparsity_per_dim': sparsity_per_dim,
            'efficiency_score': efficiency_score,
            'total_variance': total_variance,
            'mu_stats': {
                'mean': np.mean(all_mu),
                'std': np.std(all_mu),
                'min': np.min(all_mu),
                'max': np.max(all_mu),
            }
        }
        
        return rapport_lea

    def print_latent_efficiency_report(self, rapport_lea):
        """
        Imprime un rapport complet d'efficacité latente avec interprétations
        """
        print("\n" + "="*80)
        print("RAPPORT D'EFFICACITÉ LATENTE (LEA) - ANALYSE COMPLÈTE")
        print("="*80)
        
        # === SECTION 1: DIMENSIONS ACTIVES ===
        print(f"\n[1] DIMENSIONS RÉELLEMENT UTILISÉES (Active Dimensions)")
        print("-" * 80)
        
        num_active = rapport_lea['active_dimensions']
        total_dims = rapport_lea['total_dimensions']
        active_ratio = num_active / total_dims
        
        print(f"   Dimensions actives (variance > 0.01):  {num_active}/{total_dims}")
        print(f"   Active Ratio:                          {active_ratio:.2%}")
        print(f"   Dimensions inactives:                  {[i for i, active in enumerate(rapport_lea['active_dims_mask']) if not active]}")
        
        # Interprétation Active Ratio
        print(f"\n   ► Interprétation:")
        if active_ratio >= 0.8:
            print(f"      ✓ Très bon - Presque toutes les dimensions sont utilisées")
        elif active_ratio >= 0.6:
            print(f"      ⚠ Modéré - {100*(1-active_ratio):.0f}% des dimensions sont inutilisées")
        elif active_ratio >= 0.4:
            print(f"      ⚠ Risque de collapse latent - Seulement {100*active_ratio:.0f}% des dimensions actives")
        else:
            print(f"      ✗ CRITIQUE - Collapse latent massif ({100*active_ratio:.0f}% utilisées)")
        
        # === SECTION 2: VARIANCE PAR DIMENSION ===
        print(f"\n[2] DISTRIBUTION DE LA VARIANCE PAR DIMENSION")
        print("-" * 80)
        
        variance = rapport_lea['dimension_variance']
        usage = rapport_lea['usage_factor']
        active_mask = rapport_lea['active_dims_mask']
        
        # Créer un histogramme réel avec matplotlib
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Graphe 1: Variance par dimension
        colors = ['green' if active else 'red' for active in active_mask]
        ax1.bar(range(len(variance)), variance, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Dimension Latente', fontsize=12)
        ax1.set_ylabel('Variance', fontsize=12)
        ax1.set_title('Variance par Dimension Latente', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        for i, v in enumerate(variance):
            ax1.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Graphe 2: Usage factor (contribution %)
        ax2.bar(range(len(usage)), usage * 100, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Dimension Latente', fontsize=12)
        ax2.set_ylabel('Usage Factor (%)', fontsize=12)
        ax2.set_title('Contribution de Chaque Dimension (%)', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        for i, u in enumerate(usage * 100):
            ax2.text(i, u, f'{u:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # Ajouter légende
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', alpha=0.7, label='Dimension Active'),
                          Patch(facecolor='red', alpha=0.7, label='Dimension Inactive')]
        fig.legend(handles=legend_elements, loc='upper right', fontsize=11)
        
        plt.tight_layout()
        
        # Sauvegarder le graphe
        histogram_path = os.path.join(self.output_dir, 'latent_variance_histogram.png')
        plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
        print(f"\n   ✓ Histogramme sauvegardé: {histogram_path}")
        plt.close()
        
        # Afficher aussi les détails en texte
        print(f"\n   Détails variance et usage factor:\n")
        for i, (var, use) in enumerate(zip(variance, usage)):
            active = rapport_lea['active_dims_mask'][i]
            active_marker = "✓" if active else "✗"
            print(f"   Dim {i:2d} {active_marker} | var={var:8.5f}  usage={100*use:6.2f}%")
        
        # === SECTION 3: REDONDANCE ===
        print(f"\n[3] CORRÉLATION ENTRE DIMENSIONS (Redondance)")
        print("-" * 80)
        
        mean_corr = rapport_lea['mean_correlation']
        print(f"   Corrélation moyenne:  {mean_corr:.4f}")
        print(f"\n   ► Interprétation:")
        if mean_corr < 0.1:
            print(f"      ✓ Excellent - Dimensions fortement indépendantes")
        elif mean_corr < 0.2:
            print(f"      ✓ Bon - Dimensions relativement indépendantes")
        elif mean_corr < 0.5:
            print(f"      ⚠ Modéré - Redondance présente")
        else:
            print(f"      ✗ Problèmatique - Forte redondance entre dimensions")
        
        # === SECTION 4: STATISTIQUES GLOBALES ===
        print(f"\n[4] STATISTIQUES DE L'ESPACE LATENT")
        print("-" * 80)
        
        mu_stats = rapport_lea['mu_stats']
        print(f"   Moyenne des μ:                {mu_stats['mean']:10.6f}")
        print(f"   Écart-type:                   {mu_stats['std']:10.6f}")
        print(f"   Plage [min, max]:             [{mu_stats['min']:8.4f}, {mu_stats['max']:8.4f}]")
        print(f"   Variance totale:              {rapport_lea['total_variance']:10.6f}")
        
        # === SECTION 5: SCORE D'EFFICACITÉ GLOBAL ===
        print(f"\n[5] SCORE D'EFFICACITÉ GLOBAL")
        print("-" * 80)
        
        efficiency = rapport_lea['efficiency_score']
        print(f"   Score: {efficiency:.4f}/1.0")
        print(f"   Formule: (Active Ratio) × (1 - Corrélation moyenne)")
        print(f"           = ({active_ratio:.2%}) × (1 - {mean_corr:.4f})")
        print(f"           = {efficiency:.4f}")
        
        # Seuils d'interprétation
        print(f"\n   ► Interprétation:")
        if efficiency >= 0.7:
            print(f"      ✓ TRÈS BONNE - Espace latent bien utilisé et indépendant")
        elif efficiency >= 0.6:
            print(f"      ✓ BON - Utilisation acceptable de l'espace latent")
        elif efficiency >= 0.4:
            print(f"      ⚠ MODÉRÉ - Redondance ou dimensions inutilisées")
        else:
            print(f"      ✗ MAUVAIS - Collapse latent ou forte redondance")
        
        # === SECTION 6: RECOMMANDATIONS ===
        print(f"\n[6] RECOMMANDATIONS")
        print("-" * 80)
        
        issues = []
        
        if active_ratio < 0.5:
            issues.append(f"• Collapse latent: {num_active} dims actives sur {total_dims} - Réduire z_dim")
        
        if mean_corr > 0.3:
            issues.append(f"• Forte redondance: {mean_corr:.2%} corrélation moyenne - Beta trop faible?")
        
        if efficiency < 0.4:
            issues.append(f"• Efficacité très faible - Revoir hyperparamètres (beta, lambda)")
        
        if not issues:
            print("   ✓ Aucun problème détecté - Configuration optimale")
        else:
            for issue in issues:
                print(f"   {issue}")
        
        print("\n" + "="*80 + "\n")
        
        return rapport_lea

