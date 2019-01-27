import warnings
import numpy as np
import pandas as pd
from utils import (
    date, get_df, day_of_year, day_to_sine, day_length, rolling_average,
    normalise, normalise_numeric, SPECIES_MAP, closest_coord, string_time_to_minutes
)

warnings.filterwarnings('ignore')
np.random.seed(2)


def normalise_columns(df, columns=None):
    if not columns:
        columns = df.columns
    for column in columns:
        df[column] = normalise_numeric(df[column])
    return df


def prepro_train(df):

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

    df['date'] = df['date'].apply(date)
    df['day'] = df['date'].apply(day_of_year)
    df['daysine'] = df['day'].apply(day_to_sine)
    df = species_features(df)
    df = normalise_columns(df, columns=['latitude', 'longitude', 'nummosquitos'])
    return df


def prepro_weather(df):
    df = df[df.station == 1]
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df = df.replace('M', None)
    df = df.replace('T', '0.001')
    df = df.fillna(method='pad')
    df['date'] = df['date'].apply(date)
    df['year'] = df['date'].apply(lambda x: x.year)
    df['daylight'] = df.apply(lambda row: day_length(row['sunrise'], row['sunset']), axis=1)
    year_dfs = []
    for year, year_df in df.groupby('year'):
        year_df['cumtmax'] = rolling_average(year_df['tmax'], window=10)
        year_df['cumtmin'] = rolling_average(year_df['tmin'], window=10)
        year_df = year_df.fillna(method='pad')
        year_dfs.append(year_df)
    df = pd.concat(year_dfs, axis=0)
    df = normalise_columns(
        df,
        columns=[
            'sunrise', 'sunset', 'tmax', 'tmin', 'tavg', 'preciptotal', 'heat', 'cool',
            'stnpressure', 'sealevel', 'resultspeed', 'resultdir', 'avgspeed', 'snowfall'
        ]
    )
    return df


def prepro_indicators(df):
    df.drop(['community area name', 'community area number'], axis=1, inplace=True)
    df = normalise_columns(df)
    return df


class Loader(object):

    def __init__(self):
        self.target = 'wnvpresent'
        self.traps = prepro_train(get_df('train'))
        self.spray = get_df('spray')
        self.weather = prepro_weather(get_df('weather'))
        self.indicators = prepro_indicators(get_df('se_indicators'))
        self.merged = None
        self.train = None
        self.test = None

    def merge(self):
        keep_cols = [
            'year',
            'daysine',
            'latitude',
            'longitude',
            'nummosquitos',
            'species1',
            'species2',
            'species3',
            'preciptotal',
            'tmax',
            'tmin',
            'tavg',
            'daylight',
            'avgspeed',
            'resultdir',
            'resultspeed',
            'cumtmax',
            'cumtmin',
            self.target
        ]
        self.merged = pd.merge(self.traps, self.weather, on='date')
        self.merged = self.merged[keep_cols]
        self.merged.reset_index(inplace=True)

    def split(self):
        train = self.merged[self.merged.year.isin([2007, 2011])]
        test = self.merged[self.merged.year.isin([2009, 2013])]
        train.drop(['year'], axis=1, inplace=True)
        test.drop(['year'], axis=1, inplace=True)
        self.train_i = train.drop(self.target, axis=1)
        self.train_t = train[self.target]
        self.test_i = test.drop(self.target, axis=1)
        self.test_t = test[self.target]
        self.test_i['nummosquitos'] = 1


def test_train(loader):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    scores = []
    for exp in range(30):
        loader.merge()
        loader.split()
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            criterion='entropy',
        ).fit(loader.train_i, loader.train_t)
        probas = model.predict_proba(loader.test_i)[:, 1]
        scores.append(roc_auc_score(loader.test_t, probas))
        print('{exp}. ROC AUC: {score}'.format(
            exp=exp,
            score=scores[-1]
        ))
    print('Mean score: {}'.format(np.mean(scores)))


if __name__ == '__main__':
    loader = Loader()
    test_train(loader)
