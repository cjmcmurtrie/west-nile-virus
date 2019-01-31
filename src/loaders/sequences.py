import numpy as np
import pandas as pd
from src.builders.traps import build_train_traps, build_test_traps, impute_test_incidence
from src.builders.weather import build_weather, build_aggregate_weather
from src.builders.indicators import build_indicators
from src.builders.utils import normalise_columns, normalise_series


COLS = {
    'final_features': [
        # traps features
        # 'latitude', 'longitude',
        'year', 'day', 'day_sine', 'month_cosine', 'month', 'date', 'trap', 'block',
        'species1', 'species2', 'species3', 'species4', 'species5', 'species6',
        'nummosquitos', 'trap_nummosquitos_mean', 'block_nummosquitos_mean',
        # weather features
        'tmax', 'tmin', 'tavg', 'depart', 'dewpoint',
        'heat', 'cool', 'sunrise', 'sunset', 'snowfall', 'preciptotal', 'stnpressure',
        'sealevel', 'resultspeed', 'resultdir', 'avgspeed'
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
        # self.weather = build_weather()
        self.weather = build_aggregate_weather('month')
        self.indicators = build_indicators()
        self.eval = build_test_traps(self.traps)
        self._merge()
        self.data = self.traps[COLS['final_features'] + [self.target]]
        self.eval = self.eval[COLS['final_features']]
        self.data = normalise_columns(self.data)
        self.data.sort_values('date', ascending=True)
        self.eval.sort_values('date', ascending=True)
        self.eval = self.eval[COLS['final_features']]
        self.train_in = None
        self.train_tar = None
        self.test_in = None
        self.test_tar = None

    def _merge(self, merge_weather=True, merge_indicators=False):
        if merge_weather:
            self.traps = pd.merge(
                self.traps, self.weather,
                how='left', on='month', suffixes=['', '_weather']
            )
            self.eval = pd.merge(
                self.eval, self.weather,
                how='left', on='month', suffixes=['', '_weather']
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
            self.train = self.data[self.data.year.isin(train_years)]
            self.test = self.data[self.data.year.isin([year])]
        elif mode == 'submit':
            pass

    def get_train_sequences(self):
        train = self.train
        seqs = []
        tars = []
        for trap, data in train.groupby(['block', 'year']):
            seq_in = data.drop([self.target, 'date', 'trap', 'block'], axis=1)
            seq_tar = data[self.target]
            seqs.append(seq_in.apply(pd.to_numeric, errors='coerce').values)
            tars.append(seq_tar.values)
        return seqs, tars

    def get_test_sequences(self):
        test = self.test
        seqs = []
        tars = []
        for trap, data in test.groupby('block'):
            seq_in = data.drop([self.target, 'date', 'trap', 'block'], axis=1)
            seq_tar = data[self.target]
            seqs.append(seq_in.values)
            tars.append(seq_tar.values)
        return seqs, tars

    def get_eval_sequences(self):
        eval = self.eval
        for trap, data in eval.groupby('trap'):
            yield normalise_columns(data).values

    def num_columns(self):
        return len(self.train.columns) - 4      # date, trap, wnvpresent removed from inputs.


if __name__ == '__main__':

    loader = Loader()
    loader.split()
    for i, (seq_in, seq_tar) in enumerate(loader.get_train_sequences()):
        print(i)
    for i, (seq_in, seq_tar) in enumerate(loader.get_test_sequences()):
        print(i)
    for i, seq_in in enumerate(loader.get_eval_sequences()):
        print(i)
