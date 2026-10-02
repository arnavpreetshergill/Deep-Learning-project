# Multi-Modal HuffPost News Classification

This project compares four architectures on the HuffPost News Category Dataset v3:

- MLP using TF-IDF/SVD text features
- TextCNN
- BiLSTM
- Compact pretrained BERT

Every architecture is trained twice: once using text alone and once using text plus ten simple engineered metadata features.

## Run

```powershell
python run_news_classification.py --models all
```

The default uses all cleaned articles, one epoch for MLP/CNN/BiLSTM, two epochs for compact BERT, and lightweight CNN/BiLSTM layers. It is designed to finish the full eight-run comparison in under 30 minutes on the available RTX 3070 GPU.

For an even smaller development test:

```powershell
python run_news_classification.py --models mlp cnn --max-samples 12000 --epochs 1
```

The runner creates `outputs_fast_full/` containing:

- cleaned-data audit and split indices
- category, missing-value, text-length, and metadata-correlation figures
- training curves for every model
- `metrics.csv` with the final comparison
- metadata-fusion gain chart and the best model's confusion matrix
- saved PyTorch model weights and per-model classification reports

## Basic preprocessing

1. Remove exact and repeated-article duplicates.
2. Fill missing text, author, and URL values with explicit placeholders.
3. Combine `headline` and `short_description` as the text input.
4. Create ten metadata features: headline/description character and word counts, punctuation count, author count, train-only author frequency, publication year, URL length, and URL hyphen count.
5. Make one stratified 80/10/10 train/validation/test split. The scalers, TF-IDF vocabulary, SVD, and word vocabulary are fit on training data only.

The default Transformer is `google/bert_uncased_L-4_H-256_A-4`: it is a genuine pretrained BERT architecture, reduced to four layers so the full comparison can be run on an 8 GB GPU.

The compact BERT input is capped at 96 tokens and dynamically trimmed per batch, so short news briefs do not waste GPU time on padding.
