import argparse
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser(description='Beta-VAE — Détection anomalie (REAL / FAKE)')

    # ── Mode ──────────────────────────────────────────────────────────────
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'test', 'predict', 'analyze', 'compare'],
                        help='train   : entraîner le modèle\n'
                             'test    : évaluer le dossier test (REAL/FAKE)\n'
                             'predict : prédire une seule image\n'
                             'analyze : analyser l\'espace latent\n'
                             'compare : comparer net_H vs net_B')

    # ── Image unique (mode predict) ───────────────────────────────────────
    parser.add_argument('--image', type=str, default=None,
                        help='Chemin vers l\'image à prédire (mode predict uniquement)')

    # ── Données ───────────────────────────────────────────────────────────
    parser.add_argument('--train_dset_dir', type=str, default='./data/train',
                        help='Dossier des images d\'entraînement')
    parser.add_argument('--test_dset_dir', type=str, default='./data/test',
                        help='Dossier des images de test')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)  # 0 obligatoire sur Windows

    # ── Modèle ────────────────────────────────────────────────────────────
    parser.add_argument('--z_dim',           type=int,   default=16)
    parser.add_argument('--beta',            type=float, default=4.0)
    parser.add_argument('--gamma',           type=float, default=1.0)
    parser.add_argument('--C_max',           type=float, default=25.0)
    parser.add_argument('--C_stop_iter',     type=float, default=1e5)
    parser.add_argument('--objective',       type=str,   default='H', choices=['H', 'B'])
    parser.add_argument('--methode',         type=str,   default='beta sparsity',
                        choices=['basic', 'beta sparsity', 'L1 sparsity', 'both sparsity'])
    parser.add_argument('--lambda_sparsity', type=float, default=0.01)

    # ── Optimiseur ────────────────────────────────────────────────────────
    parser.add_argument('--lr',    type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)

    # ── Entraînement ──────────────────────────────────────────────────────
    parser.add_argument('--max_iter',     type=int, default=10000)
    parser.add_argument('--gather_step',  type=int, default=500)
    parser.add_argument('--display_step', type=int, default=500)
    parser.add_argument('--save_step',    type=int, default=1000)

    # ── Répertoires ───────────────────────────────────────────────────────
    parser.add_argument('--ckpt_dir',    type=str,  default='./checkpoints')
    parser.add_argument('--output_dir',  type=str,  default='./outputs')
    parser.add_argument('--viz_name',    type=str,  default='experiment')
    parser.add_argument('--ckpt_name',   type=str,  default='final')
    parser.add_argument('--save_output', type=bool, default=True)

    # ── Matériel ──────────────────────────────────────────────────────────
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())

    return parser.parse_args()


def load_single_image(image_path):
    """Charge une image et la prépare pour le modèle → shape (1, 3, 64, 64)."""
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)


def main():
    args = parse_args()
    from solver import Solver

    print("\n" + "="*60)
    print(f"   MODE : {args.mode.upper()}")
    print("="*60)
    print(f"  Train dir  : {args.train_dset_dir}")
    print(f"  Test dir   : {args.test_dset_dir}")
    print(f"  z_dim      : {args.z_dim}")
    print(f"  Methode    : {args.methode}")
    print(f"  CUDA       : {args.cuda}")
    print("="*60 + "\n")

    solver = Solver(args)

    # ─────────────────────────────────────────────────────────────────────
    # MODE TRAIN
    # ─────────────────────────────────────────────────────────────────────
    if args.mode == 'train':
        print("[INFO] Démarrage de l'entraînement...")
        solver.train()
        # → train() calcule self.threshold automatiquement

        # Sauvegarder le modèle et le threshold
        solver.save_checkpoint(args.ckpt_name)
        solver.save_threshold()

        print("\n" + "="*60)
        print("  ✅ Entraînement terminé")
        print(f"  ✅ Checkpoint : {args.ckpt_name}")
        print(f"  ✅ Threshold  : {solver.threshold:.4f}")
        print("="*60)

    # ─────────────────────────────────────────────────────────────────────
    # MODE TEST  (dossier entier → REAL / FAKE pour chaque image)
    # ─────────────────────────────────────────────────────────────────────
    elif args.mode == 'test':
        print("[INFO] Chargement du modèle...")
        solver.load_checkpoint(args.ckpt_name)
        solver.load_threshold()

        print("[INFO] Évaluation sur le dossier test...")
        results = solver.test()

        # Résumé
        nb_fake_H = sum(1 for r in results if r['label_H'] == 'FAKE')
        nb_real_H = sum(1 for r in results if r['label_H'] == 'REAL')
        nb_fake_B = sum(1 for r in results if r['label_B'] == 'FAKE')
        nb_real_B = sum(1 for r in results if r['label_B'] == 'REAL')

        print("\n" + "="*60)
        print("  RÉSUMÉ FINAL")
        print("="*60)
        print(f"  Threshold utilisé : {solver.threshold:.4f}")
        print(f"  net_H → REAL: {nb_real_H}  |  FAKE: {nb_fake_H}")
        print(f"  net_B → REAL: {nb_real_B}  |  FAKE: {nb_fake_B}")
        print("="*60)

    # ─────────────────────────────────────────────────────────────────────
    # MODE PREDICT  (une seule image → REAL / FAKE)
    # ─────────────────────────────────────────────────────────────────────
    elif args.mode == 'predict':
        if args.image is None:
            raise ValueError("❌ --image requis en mode predict\n"
                             "   Ex: python main.py --mode predict --image photo.jpg")
        if not os.path.exists(args.image):
            raise FileNotFoundError(f"❌ Image introuvable : {args.image}")

        print("[INFO] Chargement du modèle...")
        solver.load_checkpoint(args.ckpt_name)
        solver.load_threshold()

        print(f"[INFO] Image : {args.image}")
        img_tensor = load_single_image(args.image)

        result = solver.predict_single(img_tensor)

        print("\n" + "="*60)
        print("  RÉSULTAT")
        print("="*60)
        print(f"  net_H → {result['label_H']}  (score: {result['score_H']:.4f})")
        print(f"  net_B → {result['label_B']}  (score: {result['score_B']:.4f})")
        print(f"  Threshold : {solver.threshold:.4f}")
        print("="*60)

    # ─────────────────────────────────────────────────────────────────────
    # MODE ANALYZE  (analyse de l'espace latent + graphes)
    # ─────────────────────────────────────────────────────────────────────
    elif args.mode == 'analyze':
        print("[INFO] Chargement du modèle...")
        solver.load_checkpoint(args.ckpt_name)

        print("[INFO] Analyse de l'espace latent...")
        results = solver.analyze_latent_training()

        print("\n" + "="*60)
        print("  RÉSULTATS ANALYSE LATENTE")
        print("="*60)
        print(f"  Active dims     : {results['num_active_dims']}/{args.z_dim}")
        print(f"  Active ratio    : {results['active_ratio']:.4f}")
        print(f"  Efficiency      : {results['efficiency_score']:.4f}")
        print(f"  Mean corr       : {results['mean_correlation']:.4f}")
        print(f"  Total variance  : {results['total_variance']:.4f}")
        print(f"  Graphes sauvés dans : {args.output_dir}/{args.viz_name}/")
        print("="*60)

    # ─────────────────────────────────────────────────────────────────────
    # MODE COMPARE  (compare net_H vs net_B)
    # ─────────────────────────────────────────────────────────────────────
    elif args.mode == 'compare':
        print("[INFO] Chargement du modèle...")
        solver.load_checkpoint(args.ckpt_name)

        print("[INFO] Comparaison net_H vs net_B...")
        comparison = solver.compare_models()
        solver.print_comparison(comparison)
        best = solver.select_best_model(comparison)

        print("\n" + "="*60)
        print(f"  🏆 MEILLEUR MODÈLE : {best}")
        print("="*60)


if __name__ == '__main__':
    main()