"""Métriques communes à tous les modèles (baseline, BiLSTM, BiLSTM + attention, SciBERT)."""
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score, hamming_loss

GROUPES = ["cs", "math", "physics", "q-bio", "q-fin", "stat", "eess", "econ"]


def probas_vers_predictions(probas, seuils=0.5):
    """Transforme des probabilités (entre 0 et 1) en réponses oui/non (1/0).
    seuils : un seul nombre pour tous les groupes, ou une liste de 8 seuils."""
    return (np.asarray(probas) >= np.asarray(seuils)).astype(int)


def evaluer(y_vrai, y_pred):
    """y_vrai et y_pred : tableaux de 0/1 de forme (nombre d'articles, 8)."""
    return {
        "micro_f1": f1_score(y_vrai, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_vrai, y_pred, average="macro", zero_division=0),
        "hamming_loss": hamming_loss(y_vrai, y_pred),
        "subset_accuracy": accuracy_score(y_vrai, y_pred),
    }


def rapport_par_groupe(y_vrai, y_pred):
    """Précision, rappel et F1 pour chacun des 8 groupes."""
    return classification_report(y_vrai, y_pred, target_names=GROUPES, zero_division=0)



def chercher_seuils(y_vrai, probas, candidats=np.arange(0.05, 0.96, 0.05)):
    """Pour chaque groupe, essaie les seuils 0.05, 0.10, ... 0.95
    et garde celui qui donne le meilleur F1.
    À utiliser UNIQUEMENT sur la validation (jamais sur le test)."""
    y_vrai = np.asarray(y_vrai)
    probas = np.asarray(probas)
    seuils = []
    for j in range(y_vrai.shape[1]):  # j = numéro du groupe (0 à 7)
        f1s = [f1_score(y_vrai[:, j], (probas[:, j] >= s).astype(int), zero_division=0)
               for s in candidats]
        meilleur = candidats[int(np.argmax(f1s))]  # le seuil qui a donné le F1 le plus haut
        seuils.append(round(float(meilleur), 2))
    return np.array(seuils)