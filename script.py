from fastai.text.all import *
import pandas as pd
from multiprocessing import freeze_support

def main():
    # Load data
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    print(train_df['label'].value_counts())

    # Create DataLoaders
    dls = TextDataLoaders.from_df(
        train_df,                # Training DataFrame
        text_col='reviews',       # Column with text (ensure this matches your CSV!)
        label_col='label',       # Column with labels
        valid_pct=0.2,           # Validation split
        seed=42                  # Reproducibility
    )

    # Initialize classifier
    learn = text_classifier_learner(
        dls,
        AWD_LSTM,
        metrics=accuracy
    )

    # Train
    learn.fit_one_cycle(3, 1e-2)

    # Evaluate on test data
    test_dls = dls.test_dl(test_df['reviews'])
    preds, _ = learn.get_preds(dl=test_dls)

    # Example predict first 5 reviews
    for i in range(5):
        print(f"Review: {test_df['reviews'].iloc[i]}")
        print(f"Predicted: {learn.predict(test_df['reviews'].iloc[i])[0]}")
        print("-----")

if __name__ == '__main__':
    freeze_support()  # Required for Windows multiprocessing
    main()