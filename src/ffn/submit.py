import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from config import BASE_PATH
from src.models.ffn import FFN


def smooth_probas(probas, box_pts):
    box = np.ones(box_pts)/box_pts
    smooth = np.convolve(probas, box, mode='same')
    return smooth


def rescale_probas(probas, loader):
    year_incidence = {
        2008: 548.,
        2010: 1880.,
        2012: 2766.,
        2014: 900.
    }
    year_factors = {
        k: v / sum(year_incidence.values())
        for k, v in year_incidence.items()
    }
    years = loader.eval_years
    for year, factor in year_factors.items():
        probas[years[years == year].index] *= factor ** 1.25
    return probas


def build_submission(loader, plot=True):
    loader.split(mode='submit')
    train_in, train_tar = loader.get_train()
    model = FFN(n_features=loader.num_columns(), hidden=500, n_classes=1)
    model.fit(train_in, train_tar, epochs=10)
    eval_in = loader.get_eval()
    probas = model.predict_proba(eval_in)
    probas = rescale_probas(probas, loader)
    probas = probas
    if plot:
        plt.plot(probas)
        plt.show()
    ids = range(1, probas.shape[0] + 1)
    submission = pd.DataFrame({
        'Id': ids,
        'WnvPresent': probas
    })
    submission.to_csv(
        os.path.join(BASE_PATH, 'submissions/submission_{}.csv'.format(datetime.now())),
        index=False
    )
