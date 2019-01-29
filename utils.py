import os
import re
import math
import pandas as pd
from datetime import datetime
from config import BASE_PATH


SPECIES_MAP = {
    'CULEX RESTUANS': list("100000"),
    'CULEX TERRITANS': list("010000"),
    'CULEX PIPIENS': list("001000"),
    'CULEX PIPIENS/RESTUANS': list("101000"),
    'CULEX ERRATICUS': list("000100"),
    'CULEX SALINARIUS': list("000010"),
    'CULEX TARSALIS':  list("000001"),
    'UNSPECIFIED CULEX': list("001100")
}


def string_time_to_minutes(string):
    hours, minutes = (int(str(string)[:-2]), int(str(string)[-2:]))
    return (hours * 60. + minutes) // 60.


def get_df(dataset, aggregate=True):
    df = pd.read_csv(
        open(os.path.join(BASE_PATH, 'data/{}.csv'.format(dataset)), 'r')
    )
    df.columns = map(str.lower, df.columns)
    if dataset in ('train', 'test') and aggregate:
        cols = list(df.columns)
        if 'nummosquitos' in cols:
            cols.remove('nummosquitos')
            df = df.groupby(cols)['nummosquitos'].apply(sum).reset_index()
    return df


def date(text):
    return datetime.strptime(text, "%Y-%m-%d")


def day_of_year(dt):
    return dt.timetuple().tm_yday


def month_num(dt):
    return dt.month


def month_to_cosine(dt):
    return math.cos(2 * math.pi / 12 * (month_num(dt) - 8))


def day_to_sine(day):
    # see: https://math.stackexchange.com/questions/650223/formula-for-sine-wave-that-lines-up-with-calendar-seasons
    return math.sin(2 * math.pi / 365 * (day - 81.75))


def normalise(pd_series):
    return (pd_series - pd_series.min()) / (pd_series.max() - pd_series.min())


def normalise_numeric(series):
    return normalise(pd.to_numeric(series))


def day_length(sunrise, sunset):
    sunrise = (int(str(sunrise)[:-2]), int(str(sunrise)[-2:]))
    sunset = (int(str(sunset)[:-2]), int(str(sunset)[-2:]))
    day_len = (sunset[0] * 60 + sunset[1]) - (sunrise[0] * 60 + sunrise[1])
    return day_len // 60


def coordinate_distance(c1, c2):
    return math.sqrt((c1[0] - c2[0]) ** 2. + (c1[1] - c2[1]) ** 2.)


def closest_coord(c1, coords, val='index'):
    dists = []
    for i, coord in enumerate(coords):
        dists.append(coordinate_distance(c1, coord))
    if val == 'index':
        return dists.index(min(dists))
    elif val == 'distance':
        return min(dists)


def rolling_average(series, window, min_periods=1):
    return series.rolling(window=window, min_periods=min_periods).mean()


def address_zipcode(address):
    match = re.findall('\d{5,}', address)
    if match:
        return match[0]
    else:
        return None


def load_city_data():
    indicators = get_df('se_indicators')
    census = get_df('census')
    city = pd.merge(indicators, census, left_on='community area number', right_on='geogkey')
    zipcodes = get_df('comarea_zipcode')
    df = pd.merge(city, zipcodes, left_on='community area number', right_on='chgoca')
    df = df.groupby('zcta5').mean().reset_index()
    return df


def trap_incidence(df):
    df['year'] = df['date'].apply(lambda x: x.year)
    idf = get_df('train')
    idf['year'] = idf['date'].apply(lambda x: date(x).year)
    idf = idf.groupby(['year', 'trap']).sum().reset_index()
    idf = idf[['year', 'trap', 'wnvpresent']]
    idf.rename(columns={'wnvpresent': 'tincidence'}, inplace=True)
    df = pd.merge(df, idf, how='left', on=['year', 'trap'], suffixes=['', '_'])
    df.to_csv('tincidence.csv')
    return df


def infer_test_trap_incidence(df):
    traindf = get_df('train')
    traindf['date'] = traindf['date'].apply(lambda x: date(x))
    tri = trap_incidence(traindf)
    tri = tri[['year', 'trap', 'tincidence']].drop_duplicates()
    df['wnvpresent'] = None
    tei = trap_incidence(df)
    tei = tei[['year', 'trap', 'tincidence']].drop_duplicates()
    mergedi = pd.concat([tri, tei], axis=0)
    mergedi.sort_values(['trap', 'year'], inplace=True)
    dfs = []
    for trap, data in mergedi.groupby('trap'):
        data = data.interpolate()
        try:
            data = data.fillna(data['tincidence'].mode().values[0])
        except IndexError:
            data['tincidence'] = mergedi.tincidence.mean()
        dfs.append(data)
    mergedi = pd.concat(dfs, axis=0)
    mergedi.iloc[:].tincidence[mergedi.year == 2010] *= 3.1
    mergedi.iloc[:].tincidence[mergedi.year == 2012] *= 2.0
    mergedi.iloc[:].tincidence[mergedi.year == 2014] *= 0.5
    df = pd.merge(df, mergedi, how='left', on=['year', 'trap'], suffixes=['tr_', ''])
    df['tincidencebinary'] = (df['tincidence'].values > 0).astype(int)
    return df


def block_incidence(df):
    df['year'] = df['date'].apply(lambda x: x.year)
    idf = get_df('train')
    idf['year'] = idf['date'].apply(lambda x: date(x).year)
    idf = idf.groupby(['year', 'block']).sum().reset_index()
    idf = idf[['year', 'block', 'wnvpresent']]
    idf.rename(columns={'wnvpresent': 'bincidence'}, inplace=True)
    df = pd.merge(df, idf, how='left', on=['year', 'block'], suffixes=['', '_'])
    df.to_csv('bincidence.csv')
    return df


def infer_test_block_incidence(df):
    traindf = get_df('train')
    traindf['date'] = traindf['date'].apply(lambda x: date(x))
    tri = block_incidence(traindf)
    tri = tri[['year', 'block', 'bincidence']].drop_duplicates()
    df['wnvpresent'] = None
    tei = block_incidence(df)
    tei = tei[['year', 'block', 'bincidence']].drop_duplicates()
    mergedi = pd.concat([tri, tei], axis=0)
    mergedi.sort_values(['block', 'year'], inplace=True)
    dfs = []
    for block, data in mergedi.groupby('block'):
        data = data.interpolate()
        try:
            data = data.fillna(data['bincidence'].mode().values[0])
        except IndexError:
            data['bincidence'] = mergedi.bincidence.mean()
        dfs.append(data)
    mergedi = pd.concat(dfs, axis=0)
    mergedi.iloc[:].bincidence[mergedi.year == 2010] *= 3.1
    mergedi.iloc[:].bincidence[mergedi.year == 2012] *= 2.0
    mergedi.iloc[:].bincidence[mergedi.year == 2014] *= 0.5
    df = pd.merge(df, mergedi, how='left', on=['year', 'block'], suffixes=['tr_', ''])
    df['bincidencebinary'] = (df['bincidence'].values > 0).astype(int)
    return df


def zipcode_incidence():
    df = get_df('train')
    df['zipcode'] = pd.to_numeric(df['address'].apply(address_zipcode))
    incidence = df.groupby('zipcode').sum().reset_index()
    incidence = incidence[['zipcode', 'wnvpresent']]
    incidence.rename(columns={'wnvpresent': 'zincidence'}, inplace=True)
    return incidence


if __name__ == '__main__':
    train = get_df('train')
    train = trap_incidence(train)
    test = get_df('test')
    infer_test_trap_incidence(test)
