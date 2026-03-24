import argparse
import numpy as np
import torch
from solver import Solver


def parse_args():
    """Parse et retourne les arguments en ligne de commande"""
    parser = argparse.ArgumentParser(description='Beta-VAE for Anomaly Detection')
    
    # Paramètres de données
    parser.add_argument('--train_dset_dir', type=str, required=True,
                        help='Chemin vers le dossier contenant les images d\'entraînement (images réelles)')
    parser.add_argument('--test_dset_dir', type=str, required=True,
                        help='Chemin vers le dossier contenant les images de test (mélange normal + anomalies)')
    parser.add_argument('--dset_dir', type=str, default='data',
                        help='Répertoire parent des données')
    parser.add_argument('--dataset', type=str, default='anomaly_detection',
                        help='Nom du dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Taille du batch')
    parser.add_argument('--image_size', type=int, default=64,
                        help='Taille des images après redimensionnement')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Nombre de workers pour le chargement des données')
    
    # Paramètres du modèle
    parser.add_argument('--model', type=str, default='H', choices=['H', 'B'],
                        help='Architecture du modèle: H pour BetaVAE_H, B pour BetaVAE_B')
    parser.add_argument('--z_dim', type=int, default=10,
                        help='Dimension de l\'espace latent')
    
    # Paramètres d'entraînement
    parser.add_argument('--max_iter', type=int, default=1000000,
                        help='Nombre maximum d\'itérations d\'entraînement')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Coefficient beta1 pour Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Coefficient beta2 pour Adam optimizer')
    
    # Paramètres spécifiques au VAE
    parser.add_argument('--objective', type=str, default='H', choices=['H', 'B'],
                        help='Objectif d\'entraînement: H pour Beta-VAE classique, B pour Factor-VAE')
    parser.add_argument('--beta', type=float, default=4.0,
                        help='Coefficient beta pour la régularisation KL (objectif H)')
    parser.add_argument('--gamma', type=float, default=100.0,
                        help='Coefficient gamma pour la contrainte C (objectif B)')
    parser.add_argument('--C_max', type=float, default=25.0,
                        help='Valeur maximale de C pour Factor-VAE')
    parser.add_argument('--C_stop_iter', type=int, default=100000,
                        help='Nombre d\'itérations pour atteindre C_max')
    
    # Sparsité
    parser.add_argument('--methode', type=str, default='beta sparsity', 
                        choices=['beta sparsity', 'L1 sparsity', 'both sparsity'],
                        help='Méthode de sparsité pour les dimensions latentes')
    parser.add_argument('--lambda_sparsity', type=float, default=0.1,
                        help='Coefficient de sparsité L1')
   
    
    # Checkpoints et sauvegarde
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints',
                        help='Répertoire pour sauvegarder les checkpoints')
    parser.add_argument('--ckpt_name', type=str, default=None,
                        help='Nom du checkpoint à charger pour reprendre l\'entraînement')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Répertoire pour sauvegarder les résultats')
    parser.add_argument('--viz_name', type=str, default='exp1',
                        help='Nom de l\'expérience')
    parser.add_argument('--save_output', type=bool, default=True,
                        help='Sauvegarder les résultats')
    
    # Fréquences de sauvegarde et affichage
    parser.add_argument('--gather_step', type=int, default=100,
                        help='Fréquence de collecte des données de suivi (en itérations)')
    parser.add_argument('--display_step', type=int, default=100,
                        help='Fréquence d\'affichage des logs (en itérations)')
    parser.add_argument('--save_step', type=int, default=10000,
                        help='Fréquence de sauvegarde des checkpoints (en itérations)')
    
    # GPU
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(),
                        help='Utiliser GPU si disponible')
    
    # Anomaly Detection
    parser.add_argument('--anomaly_threshold_percentile', type=float, default=95.0,
                        help='Percentile pour calculer le seuil d\'anomalie')
    
    return parser.parse_args()


def main():
    """Fonction principale"""
    args = parse_args()
    
    print("="*60)
    print("Beta-VAE pour Anomaly Detection")
    print("="*60)
    print(f"Train dataset: {args.train_dset_dir}")
    print(f"Test dataset: {args.test_dset_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Model: {args.model}")
    print(f"Z dimension: {args.z_dim}")
    print(f"Objective: {args.objective}")
    print(f"Sparsity method: {args.methode}")
    print("="*60)
    
    # Créer le solver
    solver = Solver(args)
    
    # ===== ENTRAÎNEMENT =====
    print("\n[INFO] Démarrage de l'entraînement...")
    solver.train()
    
    # ===== TEST / ANOMALY DETECTION =====
    print("\n[INFO] Évaluation sur données de test...")
    test_scores = solver.test(solver.test_loader)
    
    # Calculer le seuil d'anomalie
    threshold = np.percentile(test_scores, args.anomaly_threshold_percentile)
    print(f"[INFO] Seuil d'anomalie calculé (percentile {args.anomaly_threshold_percentile}): {threshold:.4f}")
    
    # Détecter les anomalies
    print("\n[INFO] Détection des anomalies...")
    predictions = solver.detect_anomalies(test_scores, threshold)
    
    # ===== LATENT EFFICIENCY ANALYSIS (LEA) =====
    print("\n[INFO] Analyse de l'efficacité de l'espace latent (LEA)...")
    rapport_lea = solver.analyze_latent_efficiency(solver.test_loader)
    solver.print_latent_efficiency_report(rapport_lea)
    
    # ===== RÉSUMÉ FINAL =====
    print("\n" + "="*60)
    print("RÉSUMÉ DE L'EXPÉRIENCE")
    print("="*60)
    print(f"Nombre total d'images testées: {len(test_scores)}")
    print(f"Anomalies détectées: {predictions.sum()} ({100*predictions.sum()/len(predictions):.1f}%)")
    print(f"Seuil d'anomalie utilisé: {threshold:.4f}")
    print(f"Score d'efficacité latente: {rapport_lea['efficiency_score']:.4f}")
    print(f"Dimensions actives: {rapport_lea['active_dimensions']}/{rapport_lea['total_dimensions']}")
    print("="*60 + "\n")
    
    print("[✓] Pipeline complète terminée avec succès!")


if __name__ == '__main__':
    main()
