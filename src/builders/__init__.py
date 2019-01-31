import numpy as np
import pandas as pd
from src.builders.traps import build_train_traps, build_test_traps, impute_test_incidence
from src.builders.weather import build_weather
from src.builders.indicators import build_indicators
from src.builders.utils import normalise_columns, normalise_series


COLS = {
    'remove': ['address', 'addressaccuracy'],
    'final_features': [
        # traps features
        'latitude', 'longitude', 'year', 'day', 'month', 'day_sine', 'month_cosine',
        'species1', 'species2', 'species3', 'species4', 'species5', 'species6',
        'nummosquitos', 'trap_nummosquitos_mean', 'block_nummosquitos_mean',
        # 'trap_last_checked',
        # weather features
        'daylight',
        # None seem to do much, leaving for now.
        # indicator features
        # These aren't working either.
    ]
}


class Loader(object):
    years = [2007, 2009, 2011, 2013]

    def __init__(self, target='wnvpresent'):
        self.target = target
        self.traps = build_train_traps()
        self.weather = build_weather()
        self.indicators = build_indicators()
        self.eval = build_test_traps(self.traps)
        self._merge()
        self.data = self.traps[COLS['final_features'] + [self.target]]
        self.eval = self.eval[COLS['final_features']]
        self.train_in = None
        self.train_tar = None
        self.test_in = None
        self.test_tar = None

    def _merge(self, merge_weather=True, merge_indicators=True):
        if merge_weather:
            self.traps = pd.merge(
                self.traps, self.weather,
                how='left', on='date', suffixes=['', '_weather']
            )
            self.eval = pd.merge(
                self.eval, self.weather,
                how='left', on='date', suffixes=['', '_weather']
            )
        if merge_indicators:
            self.traps = pd.merge(
                self.traps, self.indicators,
                how='left', on='zipcode', suffixes=['', '_indicators']
            )
            self.eval = pd.merge(
                self.eval, self.indicators,
                how='left', on='zipcode', suffixes=['', '_indicators']
            )
            self.traps.afam.fillna(
                self.traps.vacants.mean(),
                inplace=True
            )
            self.eval.afam.fillna(
                self.traps.vacants.mean(),
                inplace=True
            )

    def split(self, mode='test', year=None):
        if mode == 'test':
            train_years = [y for y in self.years if y != year]
            train = self.data[self.data.year.isin(train_years)]
            test = self.data[self.data.year.isin([year])]
            train.drop(['year'], axis=1, inplace=True)
            test.drop(['year'], axis=1, inplace=True)
            self.train_in = train.drop(self.target, axis=1)
            self.train_tar = train[self.target]
            self.test_in = test.drop(self.target, axis=1)
            self.test_tar = test[self.target]
        elif mode == 'submit':
            train = self.data
            train.drop(['year'], axis=1, inplace=True)
            self.train_in = train.drop(self.target, axis=1)
            self.train_tar = train[self.target]
            self.eval_in = self.eval
            self.eval_years = self.eval['year']
            self.eval_in.drop(['year'], axis=1, inplace=True)
            self.eval_tar = None

    def get_train(self):
        inputs = normalise_columns(self.train_in).values
        targets = self.train_tar.values
        assert not np.isnan(inputs).any()
        assert not np.isnan(targets).any()
        return inputs, targets

    def get_test(self):
        inputs = normalise_columns(self.test_in).values
        targets = self.test_tar.values
        assert not np.isnan(inputs).any()
        assert not np.isnan(targets).any()
        return inputs, targets

    def get_eval(self):
        inputs = normalise_columns(self.eval_in).values
        assert not np.isnan(inputs).any()
        return inputs

    def num_columns(self):
        return len(self.train_in.columns)   # Year column is removed.

