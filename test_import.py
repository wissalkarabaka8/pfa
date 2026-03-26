# test_import.py
from solver import Solver, DataGather, reconstruction_loss, kl_divergence

# Vérifier que toutes les méthodes existent
methods = ['train', 'test', 'compute_loss', 'analyze_latent_training',
           'compare_models', 'print_comparison', 'select_best_model']

for m in methods:
    assert hasattr(Solver, m), f"❌ Méthode manquante : {m}"
    print(f"✅ {m} trouvée")