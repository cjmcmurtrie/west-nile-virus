import warnings
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from ffn import FFN
from utils import (
    date, get_df, day_of_year, day_to_sine, day_length,
    rolling_average, address_zipcode, load_city_data, trap_incidence, block_incidence, zipcode_incidence,
    normalise, normalise_numeric, SPECIES_MAP, string_time_to_minutes, month_num, month_to_cosine
)

warnings.filterwarnings('ignore')
np.random.seed(1)


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
        except TypeError as e:
            print(df[column])
            print('Error normalizing column:', column)
            print(str(e))
            exit()
    return df


def prepro_train(df):
    if 'nummosquitos' not in df.columns:
        df['nummosquitos'] = 1
    df['date'] = df['date'].apply(date)
    df['day'] = df['date'].apply(day_of_year)
    df['month'] = df['date'].apply(month_num)
    df['monthcos'] = df['date'].apply(month_to_cosine)
    df['daysine'] = df['day'].apply(day_to_sine)
    df['zipcode'] = pd.to_numeric(df['address'].apply(address_zipcode))
    # todo: you're including future values in incidence calculations. Compute incidence only on
    # past values.
    df = pd.merge(df, trap_incidence(), how='left', on='trap', suffixes=['', '_'])
    df['tincidencebinary'] = (df.tincidence.values > 0).astype(int)
    df = pd.merge(df, block_incidence(), how='left', on='block', suffixes=['', '_'])
    df['bincidencebinary'] = (df.bincidence.values > 0).astype(int)
    df = pd.merge(df, zipcode_incidence(), how='left', on='zipcode', suffixes=['', '_'])
    df['zincidencebinary'] = (df.zincidence.values > 0).astype(int)
    df = species_features(df)
    df = normalise_columns(df, columns=['nummosquitos', 'latitude', 'longitude'])
    df = grid_regions(df)
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


def grid_regions(df, binx=10, biny=5):
    # best at 20, 40.
    stepx = 1. / binx
    stepy = 1. / biny
    to_bin_x = lambda x: np.floor(x / stepx) * stepx
    to_bin_y = lambda y: np.floor(y / stepy) * stepy
    df['latbin'] = df.latitude.map(to_bin_y)
    df['lonbin'] = df.longitude.map(to_bin_x)
    df['region'] = list(zip(df.latbin, df.lonbin))
    df['region'] = df['region'].astype('category')
    df['region'] = df['region'].cat.rename_categories(
        {cat: 'region{}'.format(i) for i, cat in enumerate(df['region'].cat.categories)}
    )
    df = pd.concat(([df, pd.get_dummies(df['region'])]), axis=1)
    df.drop(['region'], axis=1, inplace=True)
    return df


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
    df['dl_binary'] = pd.Series((df['daylight'].values > 10).astype(int))
    df['sunrise'] = df['sunrise'].apply(string_time_to_minutes)
    df['sunset'] = df['sunset'].apply(string_time_to_minutes)
    year_dfs = []
    for year, year_df in df.groupby('year'):
        year_df['rainprev'] = (np.hstack([np.array([0] * 2), year_df['preciptotal'][:-2].values]).astype(float) > 0).astype(int)
        year_df['dpprev'] = np.hstack([np.array([0] * 1), year_df['dewpoint'][:-1].values])
        year_df['wbprev'] = np.hstack([np.array([0] * 1), year_df['wetbulb'][:-1].values])
        year_df['cumtmax'] = rolling_average(year_df['tmax'], window=11)
        year_df['cumtmin'] = rolling_average(year_df['tmin'], window=11)
        year_df['cumdir'] = rolling_average(year_df['resultdir'], window=2)
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
    cols = list(df.columns)
    cols.remove('community area name')
    cols.remove('community area number')
    df = normalise_columns(df, cols)
    return df


def merge_weather(traps, weather):
    merged = pd.merge(traps, weather, how='left', on='date')
    return merged.fillna(method='pad')


def merge_indicators(traps):
    city = load_city_data()
    merged = pd.merge(traps, city, how='left', left_on='zipcode', right_on='zcta5')
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
            'monthcos',
            'latitude_x',
            'longitude_x',
            'species1',
            'species2',
            'species3',
            'species4',
            'species5',
            'species6',
            'daylight',
            'dl_binary',
            'cumtmax',
            'resultdir',
            'cumdir',
            'rainprev',
            'dpprev',
            'wbprev',
            'tincidence',
            # 'tincidencebinary',
            'bincidence',
            # 'bincidencebinary',
            'zincidence',
            # 'zincidencebinary',
            'percent of housing crowded',
            'per capita income ',
            'total housing units',
            'not hispanic or latino, black or african american alone',
            'vacant housing units',
            self.target
        ]
        self.merged = self._merge_df(self.traps)

    def _merge_df(self, df):
        merged = merge_weather(df, self.weather)
        merged = merged.fillna(method='pad')
        merged = merge_indicators(merged)
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
        train_i.drop(['year'], axis=1, inplace=True)
        model = FFN(n_features=len(train_i.columns), hidden=200, n_classes=1)
        model.fit(train_i.values, train_t.values)
        self.keep_cols.remove(self.target)
        merged_test = self._merge_df(self.test)
        merged_test.drop(['year'], axis=1, inplace=True)
        probas = model.predict_proba(merged_test.values)
        plt.plot(probas)
        plt.show()
        ids = range(1, probas.shape[0] + 1)
        submission = pd.DataFrame({
            'Id': ids,
            'WnvPresent': probas
        })
        submission.to_csv('submissions/submission_{}.csv'.format(datetime.now()), index=False)


def test_train(loader, testing=True, submit=True):
    scores = []
    year_scores = []
    loader.merge()
    if testing:
        years = [2007, 2009, 2011, 2013]
        for year in years:
            for exp in range(5):
                loader.split(year, [2007, 2009, 2011, 2013])
                model = FFN(n_features=len(loader.train_i.columns), hidden=500, n_classes=1)
                model.fit(loader.train_i.values, loader.train_t.values)
                probas = model.predict_proba(loader.test_i.values)
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
    if submit:
        loader.build_submission()


if __name__ == '__main__':
    loader = Loader()
    test_train(loader)
