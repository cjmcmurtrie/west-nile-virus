import re
import math
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from utils import (
    date, get_df, day_of_year, day_to_sine, day_length, rolling_average, address_zipcode,
    normalise, normalise_numeric, SPECIES_MAP, closest_coord, string_time_to_minutes
)

warnings.filterwarnings('ignore')
np.random.seed(2)


def normalise_columns(df, columns=None, exclude_columns=None):
    if not columns:
        columns = df.columns
    columns = list(columns)
    if exclude_columns:
        for ec in exclude_columns:
            columns.remove(ec)
    for column in columns:
        try:
            df[column] = normalise_numeric(df[column])
        except TypeError:
            print('Error normalizing column:', column)
            exit()
    return df


def prepro_train(df):
    if 'nummosquitos' not in df.columns:
        df['nummosquitos'] = 1
    df['date'] = df['date'].apply(date)
    df['day'] = df['date'].apply(day_of_year)
    df['daysine'] = df['day'].apply(day_to_sine)
    df['zipcode'] = pd.to_numeric(df['address'].apply(address_zipcode))
    df = species_features(df)
    df = normalise_columns(df, columns=['nummosquitos', 'latitude', 'longitude'])
    city_regions(df[['latitude', 'longitude']].values)
    return df


def species_features(df):
    df['species'] = pd.DataFrame(
        data=df['species'].apply(lambda x: SPECIES_MAP[x])
    )
    species_cols = ['species{}'.format(i + 1) for i in range(6)]
    species_df = pd.DataFrame(
        data=df['species'].values.tolist(),
        columns=species_cols
    )
    df_concat = pd.concat([df, species_df], axis=1)
    return df_concat


def city_regions(coordinates, x_split=10, y_split=10):
    pass
    # todo


def prepro_weather(df):
    df = df[df.station == 1]
    df = df.replace('M', None, regex=True)
    df = df.replace('-', None, regex=True)
    df = df.replace('T', '0.00001', regex=True)
    df = df.fillna(method='pad')
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df['date'] = df['date'].apply(date)
    df['year'] = df['date'].apply(lambda x: x.year)
    df['daylight'] = df.apply(lambda row: day_length(row['sunrise'], row['sunset']), axis=1)
    df['sunrise'] = df['sunrise'].apply(string_time_to_minutes)
    df['sunset'] = df['sunset'].apply(string_time_to_minutes)
    year_dfs = []
    for year, year_df in df.groupby('year'):
        year_df['cumtmax'] = rolling_average(year_df['tmax'], window=10)
        year_df['cumtmin'] = rolling_average(year_df['tmin'], window=10)
        year_df['cumtavg'] = rolling_average(year_df['tavg'], window=5)
        year_df['cumprecip'] = rolling_average(year_df['preciptotal'], window=3)
        year_df = year_df.fillna(method='pad')
        year_dfs.append(year_df)
    df = pd.concat(year_dfs, axis=0)
    df = normalise_columns(
        df,
        columns=[
            'dewpoint', 'wetbulb',
            'sunrise', 'sunset', 'daylight', 'tmax', 'tmin', 'tavg', 'heat', 'cool', 'preciptotal',
            'stnpressure', 'sealevel', 'resultspeed', 'resultdir', 'avgspeed', 'snowfall'
        ]
    )
    return df


def prepro_indicators(df):
    # df.drop(['community area name', 'community area number'], axis=1, inplace=True)
    cols = list(df.columns)
    cols.remove('community area name')
    cols.remove('community area number')
    df = normalise_columns(df, cols)
    return df


def merge_weather(traps, weather):
    merged = pd.merge(traps, weather, how='left', on='date')
    return merged.fillna(method='pad')


def merge_indicators(traps, indicators):
    zipcodes = get_df('comarea_zipcode')
    economic = pd.merge(indicators, zipcodes, left_on='community area number', right_on='chgoca')
    economic = economic.groupby('zcta5').mean().reset_index()
    merged = pd.merge(traps, economic, how='left', left_on='zipcode', right_on='zcta5')
    return merged.fillna(method='pad')


class Loader(object):

    def __init__(self):
        '''
        Weather columns:
            # 'dewpoint', 'wetbulb',
            # 'sunrise', 'sunset', 'tmax', 'tmin', 'tavg', 'heat', 'cool', 'preciptotal',
            # 'stnpressure', 'sealevel', 'resultspeed', 'resultdir', 'avgspeed', 'snowfall',
        Indicator columns:
            # 'percent of housing crowded', 'percent households below poverty',
            # 'percent aged 16+ unemployed',
            # 'percent aged 25+ without high school diploma',
            # 'percent aged under 18 or over 64', 'per capita income ',
        '''
        self.target = 'wnvpresent'
        self.traps = prepro_train(get_df('train'))
        self.spray = get_df('spray')
        self.weather = prepro_weather(get_df('weather'))
        self.indicators = prepro_indicators(get_df('se_indicators'))
        self.test = prepro_train(get_df('test'))
        self.transfer_stations = get_df('transfer_stations')
        self.zipcodes = get_df('comarea_zipcode')
        self.merged = None
        self.train = None

    def merge(self):
        self.keep_cols = [
            'year',
            'daysine',
            'latitude',
            'longitude',
            'species1',
            'species2',
            'species3',
            'daylight',
            'magnitude', 'direction',
            self.target
        ]
        self.merged = self._merge_df(self.traps)

    def _merge_df(self, df):
        merged = merge_weather(df, self.weather)
        merged = merged.fillna(method='pad')
        merged = merged[self.keep_cols]
        merged = normalise_columns(merged, exclude_columns=['year'])
        return merged

    def split(self, year, years):
        years.remove(year)
        train = self.merged[self.merged.year.isin(years)]
        test = self.merged[self.merged.year.isin([year])]
        train.drop(['year'], axis=1, inplace=True)
        test.drop(['year'], axis=1, inplace=True)
        self.train_i = train.drop(self.target, axis=1)
        self.train_t = train[self.target]
        self.test_i = test.drop(self.target, axis=1)
        self.test_t = test[self.target]

    def build_submission(self):
        train_i = self.merged.drop(self.target, axis=1)
        train_t = self.merged[self.target]
        model = RandomForestClassifier(
            n_estimators=1000,
            max_depth=5,
            criterion='entropy',
        ).fit(train_i, train_t)
        self.keep_cols.remove(self.target)
        merged_test = self.merged = self._merge_df(self.test)
        probas = model.predict_proba(merged_test)[:, 1]
        probas = (probas - np.min(probas)) / (np.max(probas) - np.min(probas))
        ids = range(1, probas.shape[0] + 1)
        submission = pd.DataFrame({
            'Id': ids,
            'WnvPresent': probas
        })
        submission.to_csv('submissions/submission_{}.csv'.format(datetime.now()), index=False)


def test_train(loader):
    scores = []
    year_scores = []
    loader.merge()
    years = [2007, 2009, 2011, 2013]
    for year in years:
        for exp in range(5):
            loader.split(year, [2007, 2009, 2011, 2013])
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=5,
                criterion='entropy',
            ).fit(loader.train_i, loader.train_t)
            probas = model.predict_proba(loader.test_i.values)[:, 1]
            scores.append(roc_auc_score(loader.test_t.values, probas))
            print('{year}. {exp}. ROC AUC: {score}'.format(
                year=year,
                exp=exp,
                score=scores[-1]
            ))
        year_scores.append(np.mean(scores))
        print('Mean year score: {}'.format(year_scores[-1]))
        scores = []
    print('Mean score: {}'.format(np.mean(year_scores)))
    # loader.build_submission()


if __name__ == '__main__':
    loader = Loader()
    test_train(loader)
