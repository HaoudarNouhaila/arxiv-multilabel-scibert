# Classification multi-label de publications scientifiques (arXiv) — SciBERT et BiLSTM

Projet personnel de traitement automatique du langage (NLP). À partir du **titre et du résumé** d'un article arXiv,
mon modèle prédit **un ou plusieurs des 8 grands domaines** d'arXiv : informatique (cs), mathématiques (math),
physique (physics), biologie quantitative (q-bio), finance quantitative (q-fin), statistiques (stat),
ingénierie électrique (eess) et économie (econ).

J'ai construit toute la chaîne : extraction et échantillonnage des données, nettoyage, exploration,
une baseline TF-IDF, un BiLSTM (avec GloVe, puis avec attention), le fine-tuning de SciBERT,
et une évaluation finale rigoureuse. L'ensemble porte sur près de **40 000 résumés arXiv**.

**Résultat final : mon meilleur modèle (SciBERT) atteint un micro-F1 de 0,90 sur le jeu de test**
(intervalle de confiance à 95 % : 0,891 – 0,907). Je n'ai évalué le test qu'une seule fois, à la toute fin.

---

## 1. Résultats

### Comparaison des modèles (seuil 0,5, test évalué une seule fois)

| Modèle | Données d'entraînement | micro-F1 validation | **micro-F1 test** | macro-F1 test |
|---|---|---|---|---|
| TF-IDF + régression logistique | train (13 993) | 0,858 | 0,861 | 0,505 |
| TF-IDF + régression logistique | train + plus (33 666) | 0,874 | 0,872 | 0,606 |
| BiLSTM + GloVe | train (13 993) | 0,861 | 0,860 | 0,653 |
| BiLSTM + attention + GloVe | train (13 993) | 0,860 | 0,860 | 0,599 |
| SciBERT | train (13 993) | 0,895 | 0,895 | 0,756 |
| **SciBERT (modèle final)** | **train + plus (33 666)** | **0,904** | **0,900** | **0,766** |

Pour le modèle final, la *subset accuracy* (tous les groupes d'un article corrects) vaut 0,818 et la *Hamming loss* 0,030.

### Modèle final : F1 par groupe sur le test

| Groupe | Articles dans le test | F1 test |
|---|---|---|
| physics | 1 498 | 0,966 |
| cs | 974 | 0,913 |
| math | 717 | 0,886 |
| q-fin | 55 | 0,743 |
| econ | 52 | 0,701 |
| stat | 147 | 0,669 |
| q-bio | 59 | 0,667 |
| eess | 126 | 0,584 |

### Ce que je retiens

- **SciBERT domine** : +3 à +4 points de micro-F1 et +11 à +26 points de macro-F1 par rapport aux autres modèles.
  Un modèle pré-entraîné sur des textes scientifiques fait vraiment la différence, surtout pour les petits domaines.
- **Le BiLSTM fait jeu égal avec la baseline TF-IDF.** GloVe l'aide (+2 points sur la validation), mais l'attention n'apporte rien.
  Une baseline simple et bien réglée reste difficile à battre.
- **La quantité de données a été mon meilleur levier.** En ajoutant 19 673 articles au train, SciBERT a gagné
  +0,8 point sur la validation et +0,5 point sur le test, et le TF-IDF +1,1 point sur le test.
  Changer les hyperparamètres, au contraire, n'a rien apporté.
- **Le classement est le même sur la validation et sur le test.** L'écart validation → test du modèle final
  (−0,4 point) reste dans l'intervalle de confiance.

---

## 2. Données

### Source et étiquettes

- **Source** : métadonnées arXiv publiées sur Kaggle par Cornell University (`arxiv-metadata-oai-snapshot.json`),
  soit 3 252 452 articles.
- **Étiquettes** : chaque catégorie arXiv (par exemple `cs.LG`, `hep-th`, `math-ph`) est rattachée à l'un des 8 groupes.
  Ma table de correspondance couvre 38 archives, y compris les anciennes (`cmp-lg`, `alg-geom`, `chao-dyn`…).
  Toutes les archives de physique (`hep-*`, `cond-mat`, `astro-ph`, `quant-ph`, `math-ph`…) vont dans *physics*.
- **Multi-label** : un article reçoit tous les groupes de ses catégories (soumission croisée).
  En moyenne, un article a 1,21 groupe, et 82 % des articles n'en ont qu'un seul.

### Échantillon (version 2, proportionnel)

- **Tirage aléatoire au taux de 0,61 %** (20 000 / 3,25 M). J'ai remonté les plus petits groupes juste assez
  pour avoir environ 300 articles : 1,13 % pour q-fin et 1,65 % pour econ.
  J'ai écarté les résumés de moins de 30 mots. J'obtiens **19 982 articles**.
- **La répartition suit celle d'arXiv** : physics 50 %, cs 32 %, math 24 %, stat 4,9 %, eess 4,2 %, q-bio 2,0 %,
  q-fin 1,8 %, econ 1,7 %. Le déséquilibre entre le plus grand et le plus petit groupe est d'environ ×29.
- **Découpage stratifié multi-label** (*iterative stratification*, graine 42) : 70 / 15 / 15, soit
  **13 993 articles en train, 3 006 en validation et 2 983 en test**.
- **Articles d'entraînement supplémentaires (`train_plus.csv`)** : un second tirage avec les mêmes taux
  (graine 2024) donne **19 673 articles**, ajoutés **uniquement au train**. J'en ai exclu tous les articles déjà
  présents (même identifiant ou même texte) dans le train, la validation ou le test. La validation et le test restent identiques.
- **Au total : 39 655 résumés.**

### Prétraitement

- Les formules LaTeX (`$...$`, `$$...$$`) sont remplacées par le mot `formula`.
- Les commandes LaTeX, les accolades et les espaces multiples sont supprimés.
- Le texte donné aux modèles est : `titre nettoyé + ". " + résumé nettoyé`.
- Longueur en tokens SciBERT : médiane de 213. 5,2 % des textes dépassent 384 tokens et 0,7 % dépassent 512.

---

## 3. Modèles

| Modèle | Principaux réglages |
|---|---|
| **TF-IDF + régression logistique** | uni- et bigrammes, `min_df=2`, 100 000 traits, `sublinear_tf`, stop words anglais ; une régression logistique par groupe (One-vs-Rest, `C=4`) |
| **BiLSTM + GloVe** | vocabulaire du train (mots vus au moins 2 fois, 30 460 mots), 300 mots maximum ; embeddings GloVe 6B 200d (77 % du vocabulaire couvert) ; BiLSTM 2×128 + max-pooling ; dropout 0,3 ; Adam (lr 1e-3), lots de 64 ; arrêt anticipé (patience 3) |
| **BiLSTM + attention** | même BiLSTM, mais une attention additive remplace le max-pooling (permet de voir quels mots comptent) |
| **SciBERT** | `allenai/scibert_scivocab_uncased`, 384 tokens maximum, lr 2e-5, 4 epochs, warmup 10 %, weight decay 0,01, fp16, lots de 16 par GPU × 2 GPU T4 ; je garde l'epoch au meilleur micro-F1 de validation |

Tous les modèles sortent 8 probabilités indépendantes (sigmoïde). Un groupe est prédit si sa probabilité est ≥ 0,5.

---

## 4. Protocole d'évaluation

- **Métrique principale : micro-F1.** Je suis aussi le macro-F1 (chaque groupe compte autant),
  la subset accuracy et la Hamming loss.
- **J'ai fait tous les choix sur la validation** : modèle, nombre d'epochs, données, seuil.
  Le test n'est utilisé **qu'une seule fois**, dans `notebooks/08_evaluation_finale.ipynb`.
- **Seuil de 0,5 pour tous les groupes.** J'ai essayé d'optimiser un seuil par groupe, puis je l'ai vérifié
  honnêtement : seuils choisis sur une moitié de la validation, score mesuré sur l'autre moitié, 10 fois.
  Résultat : 0,9006 avec les seuils par groupe, contre 0,9035 au seuil 0,5. Les seuils « collaient » aux petits groupes
  (environ 50 exemples chacun), donc j'ai gardé 0,5.
- **Contrôles contre les fuites de données** : aucun identifiant, texte ou titre n'est commun entre le train
  (y compris `train_plus`), la validation et le test. Le vocabulaire et le TF-IDF sont appris sur le train seulement.
- **Vérification avant le test** : j'ai contrôlé que les modèles rechargés localement redonnent exactement les scores
  de validation obtenus sur Kaggle.

---

## 5. Expériences (scores de validation)

| Expérience | micro-F1 validation | Décision |
|---|---|---|
| SciBERT, 4 epochs (référence v2) | 0,8953 | — |
| 6 epochs au lieu de 4 | 0,8950 (meilleure epoch : 4) |  surapprentissage après la 4ᵉ epoch (la perte de validation remonte) |
| + 19 673 articles d'entraînement | **0,9035** |  gardé, +0,8 point |
| Ensemble SciBERT + TF-IDF | +0,07 point au mieux (estimation honnête) |  gain négligeable |
| Moyenne de 2 SciBERT | +0,03 point |  gain négligeable |
| Seuils optimisés par groupe | −0,3 point (estimation honnête) |  je garde 0,5 |

**Courbe d'apprentissage (TF-IDF, validation, seuil 0,5)** : avec 25 %, 50 %, 75 % et 100 % du train, j'obtiens
0,833 → 0,844 → 0,850 → 0,858. Le score montait encore : c'est pour cela que j'ai ajouté des données.

### Version 1 (échantillon rééquilibré) → version 2 (échantillon proportionnel)

- **Version 1** : j'ai d'abord travaillé sur un échantillon *rééquilibré* (au moins 2 500 articles par groupe,
  soit 17 349 articles). Les petits groupes y sont sur-représentés, la tâche est plus difficile et le micro-F1 plus bas.
  Sur la validation, avec les seuils : TF-IDF 0,792 ; BiLSTM 0,786 ; attention 0,781 ; SciBERT 0,825 ;
  SciBERT + titre 0,831. L'ajout du titre a été conservé pour la version 2.
- **Version 2** : un échantillon *proportionnel*, qui suit la répartition réelle d'arXiv. Ce cadre est plus réaliste,
  mais le micro-F1 y est dominé par physics, cs et math, qui sont plus faciles.
- **Les scores des deux versions ne sont pas comparables** : la répartition des étiquettes change.
  J'ai sauvegardé la version 1 telle quelle dans `version1_reequilibree/`.

---

## 6. Analyse des erreurs (modèle final, test)

- **81,8 % des articles** ont exactement les bons groupes. Aucun article ne reste sans groupe prédit.
- **Les petits groupes sont les plus faibles** (eess, q-bio, stat, econ) : leur F1 va de 0,58 à 0,70,
  et le rappel y est souvent inférieur à la précision (le modèle oublie plus qu'il n'invente).
- **Confusions les plus fréquentes** : cs ↔ physics, math → cs, physics → math, stat → eess, math → eess.
  Il s'agit surtout d'articles interdisciplinaires (par exemple réseaux de capteurs ou équations aux dérivées partielles),
  dont la soumission croisée est un choix des auteurs, souvent discutable même pour un humain.

---

## 7. Limites et pistes

- **Petits groupes** : le test ne contient que 52 à 147 articles pour q-bio, q-fin, stat, eess et econ,
  donc leurs scores sont instables (quelques articles font varier le F1 de plusieurs points).
- **Étiquettes bruitées** : les catégories croisées sont choisies par les auteurs, sans règle stricte.
- **Un seul entraînement par réglage** : sur GPU, trois entraînements identiques de SciBERT ont varié de 0,04 point
  au seuil 0,5 (observé en version 1). L'aléa est donc faible, mais pas nul.
- **Optimisme de la validation** : beaucoup de choix ont été faits sur la validation (0,904), ce qui explique en partie
  l'écart avec le test (0,900).
- **Comparaison pas tout à fait égale** : le BiLSTM n'a pas été réentraîné avec `train_plus`. SciBERT tronque
  5 % des textes à 384 tokens.
- **Pistes** : encore plus de données d'entraînement (la tendance restait positive), pondérer les petits groupes
  dans la fonction de perte, essayer 512 tokens.

---

## 8. Structure du projet

```
projet_nlp/
├── data/
│   ├── raw/arxiv_echantillon.csv        # échantillon v2 (19 982 articles, texte brut)
│   └── processed/                       # train.csv, val.csv, test.csv, train_plus.csv (texte nettoyé + 8 labels)
├── src/
│   ├── metrics.py                       # métriques communes (micro/macro-F1, rapport par groupe, seuils)
│   └── check_env.py                     # vérifie l'installation (PyTorch, GPU, tokenizer SciBERT)
├── notebooks/                           # notebooks exécutés en local (CPU)
│   ├── 01_exploration.ipynb             # répartition, co-occurrences, longueurs
│   ├── 02_pretraitement.ipynb           # nettoyage + découpage stratifié 70/15/15
│   ├── 03_baseline.ipynb                # TF-IDF + régression logistique
│   └── 08_evaluation_finale.ipynb       # évaluation UNIQUE sur le test + analyse des erreurs
├── notebooks_kaggle/                    # notebooks exécutés sur Kaggle (GPU T4 x2)
│   ├── v2-01-echantillon.ipynb          # tirage de l'échantillon proportionnel
│   ├── v2-02-train-plus.ipynb           # tirage des 19 673 articles d'entraînement en plus
│   ├── v2-04b-bilstm.ipynb              # BiLSTM (+ GloVe, + attention)
│   ├── v2-05-scibert.ipynb              # SciBERT (référence)
│   ├── v2-06-scibert-optim.ipynb        # essai 6 epochs
│   └── v2-07-scibert-plus-donnees.ipynb # SciBERT final (train + plus)
├── models/                              # bilstm.pt, attention.pt, bilstm_vocab.json
├── results/                             # scores (json/csv), probabilités SciBERT, figures
├── version1_reequilibree/               # sauvegarde complète de la version 1
└── requirements.txt
```

SciBERT pèse environ 440 Mo, donc ses poids ne sont pas inclus. Ses probabilités sur la validation et le test,
elles, sont dans `results/`. Elles suffisent pour refaire l'évaluation finale.

---

## 9. Reproduire

1. **Environnement local** : Python 3.11, puis `pip install -r requirements.txt`.
   Versions utilisées : torch 2.15 (CPU), transformers 5.23, datasets 5.2, scikit-learn 1.9, pandas 3.1, iterative-stratification 0.1.9.
2. **Évaluation finale seule** : les données, les modèles BiLSTM et les probabilités SciBERT sont inclus.
   Il suffit d'exécuter `notebooks/08_evaluation_finale.ipynb`.
3. **Tout refaire depuis le début** :
   1. Sur Kaggle, avec le dataset arXiv de Cornell University : `v2-01-echantillon`, puis placer le fichier obtenu dans `data/raw/`.
   2. En local : `01_exploration` → `02_pretraitement` → `03_baseline`.
   3. Sur Kaggle, GPU T4 x2, avec un dataset contenant `train/val/test.csv` et `src/metrics.py`
      (plus GloVe 6B pour le BiLSTM) : `v2-02-train-plus`, `v2-04b-bilstm`, `v2-05-scibert`, `v2-07-scibert-plus-donnees`.
      Ranger ensuite les sorties dans `data/processed/`, `models/` et `results/`.
   4. En local : `08_evaluation_finale`.

Durées indicatives sur 2 GPU T4 : SciBERT environ 36 minutes (train seul) et environ 1 h 40 (train + plus) ;
BiLSTM environ 20 minutes (4 variantes entraînées).
