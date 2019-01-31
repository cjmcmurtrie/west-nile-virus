import math
import numpy as np
import pandas as pd
from src.builders.utils import (
    write_csv, verbose, normalise_columns,
    get_df, str_to_date, day, week, month, year, address_zipcode
)


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
    cols = df.columns.tolist()
    cols.remove('nummosquitos')
    df = df.groupby(cols)['nummosquitos'].apply(sum).reset_index()
    df = prepare_traps(df)
    df = compute_incidence_cols(df)
    df = extra_features(df)
    return df


def build_test_traps(train_data):
    df = get_df('test')
    cols = df.columns.tolist()
    dup_count = df.groupby(['date', 'species', 'trap'])
    dup_count = dup_count.size().reset_index().rename(columns={0: 'count'})
    df = pd.merge(
        df, dup_count,
        how='left',
        on=['date', 'species', 'trap'],
        suffixes=['', '_']
    )
    df = df[cols + ['count']]
    df['count'] = pd.Series(
        ((df['count'].values > 1) *
         df['count'].values)
    )
    df['nummosquitos'] = df['count'] * 50
    df = prepare_traps(df)
    df = impute_test_incidence(
        df, train_data,
        obj_cols=('trap', 'block'),
        agg_cols=('wnvpresent', 'nummosquitos')
    )
    df['nummosquitos'] += df['trap_nummosquitos_mean'] * df['nummosquitos'] > 0
    df = extra_features(df)
    return df


def impute_test_incidence(df, imputation_source, obj_cols, agg_cols):
    all_data = pd.concat([df, imputation_source], sort=True)
    for obj_col in obj_cols:
        for agg_col in agg_cols:
            incidence_col = incidence_name(obj_col, agg_col)
            agg = all_data[['year', obj_col, incidence_col]].drop_duplicates()
            agg.sort_values(
                ['year', obj_col],
                ascending=True,
                inplace=True
            )
            agg.iloc[:][incidence_col][agg.year.isin([2008, 2010, 2012, 2014])] = None
            interps = []
            for value, data in agg.groupby(obj_col):
                data.iloc[:][incidence_col] = data[incidence_col].interpolate()
                interps.append(data)
            agg = pd.concat(interps)
            df[incidence_col] = pd.merge(
                df, agg,
                how='left',
                on=['year', obj_col],
                suffixes=['_', '']
            )[incidence_col]
            df[incidence_col].fillna(0, inplace=True)
            # todo: missing values are just set to zero now.
    return df


def extra_features(df):
    df['day_sine'] = df['day'].apply(day_to_sine)
    df['month_cosine'] = df['month'].apply(month_to_cosine)
    df['trap_wnv_bin'] = (df[incidence_name('trap', 'wnvpresent')] > 0).astype(int)
    df['block_wnv_bin'] = (df[incidence_name('block', 'wnvpresent')] > 0).astype(int)
    df['trap_nm_bin'] = (df[incidence_name('block', 'nummosquitos')] > 0).astype(int)
    df['block_nm_bin'] = (df[incidence_name('block', 'nummosquitos')] > 0).astype(int)
    df = trap_last_checked(df)
    return df


def trap_last_checked(df, maximum=50):
    trap_dfs = []
    for trap, data in df.groupby('trap'):
        datecol = data['date']
        diffcol = datecol.diff().apply(lambda x: x.days)
        repeat_date = datecol[1:].values == datecol[:-1].values
        repeat_date = np.insert(repeat_date, 0, False)
        mask = np.hstack([repeat_date[1:], [False]])
        diffcol[repeat_date] = diffcol[mask].values
        data['trap_last_checked'] = diffcol.values
        trap_dfs.append(data)
    new_df = pd.concat(trap_dfs)
    new_df['trap_last_checked'] = new_df['trap_last_checked'].fillna(0)
    new_df['trap_last_checked'].mask(new_df['trap_last_checked'] > maximum, maximum, inplace=True)
    new_df.sort_index(axis=0, inplace=True)
    return new_df


def day_to_sine(day):
    return math.sin(2 * math.pi / 365 * (day - 81.75))


def month_to_cosine(month):
    return math.cos(2 * math.pi / 12 * (month - 8))


def compute_incidence_cols(df):
    df = compute_incidence(df, obj_col='trap', agg_col='wnvpresent')
    df = compute_incidence(df, obj_col='block', agg_col='wnvpresent')
    df = compute_incidence(df, obj_col='trap', agg_col='nummosquitos')
    df = compute_incidence(df, obj_col='block', agg_col='nummosquitos')
    return df


def compute_incidence(df, obj_col, agg_col):
    agg_df = df.groupby(['year', obj_col]).mean().reset_index()
    agg_name = incidence_name(obj_col, agg_col)
    df = pd.merge(
        df, agg_df[['year', obj_col, agg_col]],
        how='left',
        on=['year', obj_col],
        suffixes=['', '_right']
    )
    df.rename(
        columns={agg_col + '_right': agg_name},
        inplace=True
    )
    return df


def incidence_name(obj_col, agg_col):
    return '{}_{}_mean'.format(obj_col, agg_col)


def prepare_traps(df):
    df['date'] = df['date'].apply(str_to_date)
    df['day'] = df['date'].apply(day)
    df['week'] = df['date'].apply(week)
    df['month'] = df['date'].apply(month)
    df['year'] = df['date'].apply(year)
    df['zipcode'] = pd.to_numeric(df['address'].apply(address_zipcode))
    df = species_features(df)
    # todo: add street name column.
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


def block_features(df):
    df = pd.concat([df, pd.get_dummies(df['block'], prefix='blck')], axis=1)
    return df


def aggregate_column(df, column, agg_func=sum):
    cols = list(df.columns)
    if column in cols:
        cols.remove(column)
        df = df.groupby(cols)[column].apply(agg_func).reset_index()
    return df


if __name__ == '__main__':
    train_traps = build_train_traps()
    test_traps = build_test_traps()
    test_traps = impute_test_incidence(
        test_traps, train_traps,
        obj_cols=('trap', 'block'),
        agg_cols=('wnvpresent', 'nummosquitos')
    )

