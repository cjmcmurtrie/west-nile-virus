import warnings
import numpy as np
from sklearn.metrics import roc_auc_score
from src.models.ffn import FFN
from src.loaders.vectors import Loader
from src.ffn.submit import build_submission


warnings.filterwarnings("ignore")
np.random.seed(1)


def run(test=False, submit=True):
    scores = []
    year_scores = []
    loader = Loader()
    if test:
        years = [2007, 2009, 2011, 2013]
        for year in years:
            for exp in range(5):
                loader.split(mode='test', year=year)
                model = FFN(n_features=loader.num_columns(), hidden=500, n_classes=1)
                x, y = loader.get_train()
                model.fit(x, y, epochs=10)
                xt, yt = loader.get_test()
                probas = model.predict_proba(xt)
                scores.append(roc_auc_score(yt, probas))
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
