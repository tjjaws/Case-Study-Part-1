# Sentiment Analysis of Drug Reviews (Case Study Part 1)

Classifies patient drug reviews from Drugs.com as negative, neutral, or positive. Compares three TF-IDF models (Naive Bayes, Linear SVM, Logistic Regression) with a fine-tuned DistilBERT Transformer.

## Dataset

Drug Reviews (Drugs.com), UCI Machine Learning Repository (Grasser et al., 2018). The data is not included in this repository; see the source for its terms of use. The notebook downloads the Kaggle mirror automatically with kagglehub (jessicali9530/kuc-hackathon-winter-2018, file drugsComTrain_raw.csv).

Sentiment labels are derived from the 1 to 10 patient rating: 1 to 4 negative, 5 to 6 neutral, 7 to 10 positive. After removing duplicate reviews (112,329 remain), a stratified sample of 30,000 reviews is used.

## Project structure

- Case_Study_Part_1_TPJ.ipynb: full pipeline (loading, preprocessing, features, models, evaluation, interpretation)
- requirements.txt: library versions
- results/metrics.csv: test set metrics for all models
- results/sentiment_by_condition.csv: sentiment share by condition
- figures/: rating distribution, sentiment distribution, confusion matrices

## How to run

1. Open Case_Study_Part_1_TPJ.ipynb in Google Colab.
2. Runtime, Change runtime type, select a T4 GPU (needed for DistilBERT; the other models run on CPU).
3. Runtime, Run all. The full run takes about 15 minutes.
4. Metrics are written to results/ and plots to figures/.

To run locally instead: pip install -r requirements.txt, then run the notebook (a GPU is strongly recommended for DistilBERT).

## Pipeline

1. Preprocessing: HTML decoding, lowercasing, URL and digit removal, tokenization, stop word removal (negations kept), lemmatization.
2. Features: TF-IDF with unigrams and bigrams (30,000 features, min_df 3) for the classical models; DistilBERT tokenizer on raw text.
3. Models: Multinomial Naive Bayes, Linear SVM, Logistic Regression (5-fold grid search on macro-F1, balanced class weights where supported), DistilBERT (2 epochs, learning rate 2e-5, batch size 32, class-weighted loss).
4. Evaluation: accuracy, precision, recall, F1, macro-F1, confusion matrices on a fixed stratified 80/20 split.

## Results (test set, 6,000 reviews)

| Model | Accuracy | Macro-F1 | Neutral F1 |
| --- | --- | --- | --- |
| Naive Bayes | 0.761 | 0.487 | 0.015 |
| Linear SVM | 0.756 | 0.563 | 0.171 |
| Logistic Regression | 0.718 | 0.575 | 0.238 |
| DistilBERT | 0.768 | 0.651 | 0.323 |

## Reproducibility

Random seeds are set to 42. The classical models give identical results on every run. DistilBERT training on a GPU is not exactly repeatable, so its metrics can vary by about 0.005 between runs. Library versions are in requirements.txt.

## Reference

Grasser, F., Kallumadi, S., Malberg, H., & Zaunseder, S. (2018). Aspect-based sentiment analysis of drug reviews applying cross-domain and cross-data learning. Proceedings of the 2018 International Conference on Digital Health.
