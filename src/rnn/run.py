import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from src.models.rnn import LSTMClassifier
from src.loaders.sequences import Loader
from src.ffn.submit import build_submission


warnings.filterwarnings("ignore")
np.random.seed(1)


def run(test=True, submit=False, plot=True):
    scores = []
    year_scores = []
    loader = Loader()
    if test:
        years = [2007, 2009, 2011, 2013]
        for year in years:
            for exp in range(1):
                loader.split(mode='test', year=year)
                model = LSTMClassifier(
                    n_features=loader.num_columns(),
                    hidden=300,
                    n_classes=1
                )
                x, y = loader.get_train_sequences()
                model.fit(x, y, epochs=40, verbose=False)
                xt, yt = loader.get_test_sequences()
                probas = model.predict_proba(xt)
                if plot:
                    plt.plot(probas)
                    plt.show()
                scores.append(roc_auc_score(np.concatenate(yt), 1 - probas))
                print('{year}. {exp}. ROC AUC: {score}'.format(
                    year=year,
                    exp=exp,
                    score=scores[-1]
                ))
            year_scores.append(np.mean(scores))
            print('Mean year score: {}'.format(year_scores[-1]))
            scores = []
        print('Mean score: {}'.format(np.mean(year_scores)))
    if submit:
        build_submission(loader)


if __name__ == '__main__':
    run()
