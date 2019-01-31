import numpy as np
import pandas as pd
from src.builders.traps import build_train_traps, build_test_traps
from src.builders.weather import build_weather, build_aggregate_weather
from src.builders.indicators import build_indicators
from src.builders.utils import normalise_columns


FEATURES = {
    'trap_features': [
        'species1', 'species2', 'species3', 'species4', 'species5', 'species6',
        'latitude', 'longitude', 'year', 'day', 'week', 'month', 'day_sine', 'month_cosine',
        'nummosquitos', 'trap_nummosquitos_mean', 'block_nummosquitos_mean', 'trap_last_checked',
    ],
    'weather_features': [
        'tmax', 'tmin', 'heat', 'cool', 'dewpoint', 'preciptotal',
        'daylength', 'date', 'tmax', 'tmin', 'tavg', 'depart', 'dewpoint',
        'heat', 'cool', 'sunrise', 'sunset', 'snowfall', 'preciptotal', 'stnpressure',
        'sealevel', 'resultspeed', 'resultdir', 'avgspeed'
    ],
    'indicator_features': [
        'zipcode', 'ind_lat', 'ind_lon', 'crowding', 'income pc',
        'population', 'white', 'afam', 'asian', 'hispanic', 'total housing', 'vacants',
    ]
}


class Loader(object):
    years = [2007, 2009, 2011, 2013]

    def __init__(self, target='wnvpresent', traps=True, weather=False, indicators=False):
        self.target = target
        self.traps = build_train_traps()
        self.weather = None
        self.indicators = None
        self.eval = build_test_traps(self.traps)
        self._merge(weather=weather, indicators=indicators)
        self._select(traps=traps, weather=weather, indicators=indicators)
        self._normalize_eval()
        self.train_in = None
        self.train_tar = None
        self.test_in = None
        self.test_tar = None

    def _merge(self, weather, indicators):
        self.weather = build_aggregate_weather('month')
        if weather:
            self.traps = pd.merge(
                self.traps, self.weather,
                how='left', on='year', suffixes=['', '_weather']
            )
            self.eval = pd.merge(
                self.eval, self.weather,
                how='left', on='year', suffixes=['', '_weather']
            )
        if indicators:
            self.indicators = build_indicators()
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

    def _select(self, traps, weather, indicators):
        cols = []
        if traps:
            cols.extend(FEATURES['trap_features'])
        if weather:
            for wf in FEATURES['weather_features']:
                for wc in self.weather.columns:
                    if wf in wc:
                        cols.append(wc)
        if indicators:
            cols.extend(FEATURES['indicator_features'])
        self.data = self.traps[cols + [self.target]]
        self.eval = self.eval[cols]

    def _normalize_eval(self):
        # todo: normalize eval and training data together.
        conc = pd.concat([self.data, self.eval])
        conc = normalise_columns(conc)
        split_ix = len(self.data.index)
        self.data = conc.iloc[:split_ix][self.data.columns]
        self.eval = conc.iloc[split_ix:][self.eval.columns]

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
        inputs = self.train_in.values
        targets = self.train_tar.values
        assert not np.isnan(inputs).any()
        assert not np.isnan(targets).any()
        return inputs, targets

    def get_test(self):
        inputs = self.test_in.values
        targets = self.test_tar.values
        assert not np.isnan(inputs).any()
        assert not np.isnan(targets).any()
        return inputs, targets

    def get_eval(self):
        inputs = self.eval_in.values
        assert not np.isnan(inputs).any()
        return inputs

    def num_columns(self):
        return len(self.train_in.columns)   # Year column is removed.
