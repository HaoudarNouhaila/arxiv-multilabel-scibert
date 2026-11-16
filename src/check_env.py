# Vérifie que l'environnement est prêt : PyTorch + tokenizer SciBERT
import torch
from transformers import AutoTokenizer

# 1. PyTorch
print("Version de PyTorch :", torch.__version__)
print("GPU disponible     :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Nom du GPU         :", torch.cuda.get_device_name(0))

# 2. Tokenizer SciBERT
nom_modele = "allenai/scibert_scivocab_uncased"
tokenizer = AutoTokenizer.from_pretrained(nom_modele)
print("Tokenizer chargé   :", nom_modele)

# 3. Tokeniser une phrase scientifique
phrase = "We propose a transformer-based model for multi-label classification of scientific abstracts."
tokens = tokenizer.tokenize(phrase)
print("Tokens             :", tokens)
print("Nombre de tokens   :", len(tokens))