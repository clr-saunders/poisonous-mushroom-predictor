import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import click
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.make_pipeline import create_model_pipeline
from matplotlib import pyplot as plt
import pickle

@click.command()
@click.argument('train_df')
@click.argument('test_df')

def main(train_df, test_df):

    train_df = pd.read_csv(train_df)
    test_df = pd.read_csv(test_df)

    x_train = train_df.drop('is_poisonous', axis=1)
    y_train = train_df['is_poisonous']
    x_test = test_df.drop('is_poisonous', axis=1)
    y_test = test_df['is_poisonous']

    preprocessor = OneHotEncoder(handle_unknown="ignore")

    # dc_pipe = make_pipeline(
    #     preprocessor, 
    #     DummyClassifier(random_state=123)
    # )

    dc_pipe = create_model_pipeline(
        DummyClassifier(random_state=123),
        preprocessor
    )

    os.makedirs('scripts/models', exist_ok=True)

    dc_pipe.fit(x_train, y_train)

    with open('scripts/models/dummy.pkl', 'wb')  as f:
        pickle.dump(dc_pipe, f)

    cross_val_dc = pd.DataFrame(
        cross_validate(
            dc_pipe, x_train, y_train, cv=10, n_jobs=-1, return_train_score=True
        )).mean().to_frame().rename(columns={0: "mean_value"})

    svc_pipe = create_model_pipeline(
        SVC(random_state=123),
        preprocessor
    )

    svc_pipe.fit(x_train, y_train)

    with open('scripts/models/svc.pkl', 'wb')  as f:
        pickle.dump(svc_pipe, f)

    cross_val_svc = pd.DataFrame(
        cross_validate(
            svc_pipe, x_train, y_train, cv=10, n_jobs=-1, return_train_score=True
        )).mean().to_frame().rename(columns={0: "mean_value"})


    y_pred = svc_pipe.predict(x_test)

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(cm)
    disp.plot()

    disp.figure_.savefig("img/confusion_matrix.png")
    
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]

    precision = TP / (FP + TP) if (FP + TP) > 0 else 0
    recall    = TP / (FN + TP) if (FN + TP) > 0 else 0
    accuracy  = (TP + TN) / (TP + TN + FP + FN)

    scores_df = pd.DataFrame([{"precision": precision, "recall": recall, "accuracy": accuracy}])
    os.makedirs("results/tables", exist_ok=True)
    scores_df.to_csv("results/tables/test_scores.csv", index=False)

if __name__ == '__main__':
    main()