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
import os
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

    dc_pipe = make_pipeline(
        preprocessor, 
        DummyClassifier(random_state=123)
    )

    os.makedirs('scripts/models', exist_ok=True)

    dc_pipe.fit(x_train, y_train)

    with open('scripts/models/dummy.pkl', 'wb')  as f:
        pickle.dump(dc_pipe, f)

    cross_val_dc = pd.DataFrame(
        cross_validate(
            dc_pipe, x_train, y_train, cv=10, n_jobs=-1, return_train_score=True
        )).mean().to_frame().rename(columns={0: "mean_value"})

    svc_pipe = make_pipeline(
        preprocessor, 
        SVC(random_state=123)
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


if __name__ == '__main__':
    main()