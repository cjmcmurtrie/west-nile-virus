import os
import math
import numpy as np
import pandas as pd
from datetime import datetime
from config import BASE_PATH
from src.builders.utils import write_csv, verbose


COLS = {
    'remove': ['address', 'addressaccuracy'],
    'final_features': []
}


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


def build_train_traps():
    df = get_df('train')
    df = prepare_traps(df)
    df = compute_incidence_cols(df)
    # df = extra_features(df)
    return df


def build_test_traps():
    df = get_df('test')
    df = prepare_traps(df)
    # df = extra_features(df)
    return df


def impute_test_incidence(df, imputation_source, cols=('trap', 'block')):
    all_data = pd.concat([df, imputation_source], sort=False)
    for col in cols:
        incidence_col = '{}_incidence'.format(col)
        agg = all_data.groupby(['year', col]) \
            .mean() \
            .reset_index()[['year', col, incidence_col]]
        agg.iloc[:][incidence_col][agg.year.isin([2008, 2010, 2012, 2014])] = None
        interps = []
        for value, data in agg.groupby(col):
            data.iloc[:][incidence_col] = data[incidence_col].interpolate()
            interps.append(data)
        agg = pd.concat(interps)
        df[incidence_col] = pd.merge(
            df, agg,
            how='left',
            on=['year', col],
            suffixes=['_', '']
        )[incidence_col]
        # todo: missing values are just set to zero now.
    df.to_csv('test_inspect.csv')
    return df


def extra_features(df):
    df['day_sine'] = df['day'].apply(day_to_sine)
    df['month_cosine'] = df['month'].apply(month_to_cosine)
    return df


def day_to_sine(day):
    return math.sin(2 * math.pi / 365 * (day - 81.75))


def month_to_cosine(month):
    return math.cos(2 * math.pi / 12 * (month - 8))


def compute_incidence_cols(df):
    df = compute_incidence(df, 'trap')
    df = compute_incidence(df, 'block')
    return df


def compute_incidence(df, agg_column):
    agg_df = df.groupby(['year', agg_column]).sum().reset_index()
    df = pd.merge(
        df, agg_df[['year', agg_column, 'wnvpresent']],
        how='left',
        on=['year', agg_column],
        suffixes=['', '{}_merge_'.format(agg_column)]
    )
    df.rename(
        columns={'wnvpresent{}_merge_'.format(agg_column): '{}_incidence'.format(agg_column)},
        inplace=True
    )
    return df


def prepare_traps(df):
    df.drop(COLS['remove'], axis=1, inplace=True)
    df['date'] = df['date'].apply(str_to_date)
    df['day'] = df['date'].apply(day)
    df['month'] = df['date'].apply(month)
    df['year'] = df['date'].apply(year)
    df = species_features(df)
    # todo: add zipcode column, street name column.
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
    df = pd.concat([df, species_df], axis=1)
    df.drop('species', axis=1, inplace=True)
    return df


def get_df(dataset):
    df = pd.read_csv(
        open(os.path.join(BASE_PATH, 'data/{}.csv'.format(dataset)), 'r')
    )
    df.columns = map(str.lower, df.columns)
    return df


def aggregate_column(df, column, agg_func=sum):
    cols = list(df.columns)
    if column in cols:
        cols.remove(column)
        df = df.groupby(cols)[column].apply(agg_func).reset_index()
    return df


def str_to_date(text):
    return datetime.strptime(text, "%Y-%m-%d")


def day(dt):
    return dt.timetuple().tm_yday


def month(dt):
    return dt.month


def year(dt):
    return dt.year


if __name__ == '__main__':
    train_traps = build_train_traps()
    train_traps.to_csv('train_inspect.csv')
    test_traps = build_test_traps()
    test_traps = impute_test_incidence(test_traps, train_traps)
