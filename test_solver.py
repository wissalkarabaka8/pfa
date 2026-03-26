# test_solver.py
import torch
import argparse
from solver import Solver

if __name__ == '__main__':  # ✅ OBLIGATOIRE sur Windows (multiprocessing)

    # Créer des args fictifs
    args = argparse.Namespace(
        cuda=False,
        max_iter=10,
        z_dim=16,
        beta=4.0,
        gamma=1.0,
        C_max=25,
        C_stop_iter=1e5,
        objective='H',
        methode='beta sparsity',
        lr=1e-4,
        beta1=0.9,
        beta2=0.999,
        lambda_sparsity=0.01,
        batch_size=4,
        gather_step=5,
        display_step=5,
        save_step=5,
        save_output=True,
        ckpt_dir='./checkpoints',
        output_dir='./outputs',
        viz_name='test_run',
        train_dset_dir='./data/train',
        test_dset_dir='./data/test',
    )

    print("=" * 50)
    print("✅ [1/6] Initialisation du Solver...")
    solver = Solver(args)
    print("✅ Solver initialisé avec succès")

    print("=" * 50)
    print("✅ [2/6] Test train()...")
    solver.train()
    print("✅ train() terminé")

    print("=" * 50)
    print("✅ [3/6] Test compute_threshold()...")
    threshold = solver.compute_threshold()
    print(f"✅ Threshold: {threshold:.4f}")

    print("=" * 50)
    print("✅ [4/6] Test test()...")
    scores_H, scores_B = solver.test()
    print(f"✅ scores_H shape: {scores_H.shape}")
    print(f"✅ scores_B shape: {scores_B.shape}")

    print("=" * 50)
    print("✅ [5/6] Test analyze_latent_training()...")
    results = solver.analyze_latent_training()
    print(f"✅ Active ratio : {results['active_ratio']:.4f}")
    print(f"✅ Efficiency   : {results['efficiency_score']:.4f}")
    print(f"✅ Mean corr    : {results['mean_correlation']:.4f}")

    print("=" * 50)
    print("✅ [6/6] Test compare_models()...")
    comparison = solver.compare_models()
    solver.print_comparison(comparison)
    best = solver.select_best_model(comparison)
    print(f"✅ Best model: {best}")

    print("=" * 50)
    print("🎉 TOUS LES TESTS PASSÉS AVEC SUCCÈS")