import os
import re
import pandas as pd
from datetime import datetime
from config import BASE_PATH


def get_df(dataset):
    df = pd.read_csv(
        open(os.path.join(BASE_PATH, 'data/{}.csv'.format(dataset)), 'r')
    )
    df.columns = map(str.lower, df.columns)
    return df


def str_to_date(text):
    return datetime.strptime(text, "%Y-%m-%d")


def day(dt):
    return dt.timetuple().tm_yday


def week(dt):
    return dt.isocalendar()[1]


def month(dt):
    return dt.month


def year(dt):
    return dt.year


def string_time_to_minutes(string):
    hours, minutes = (int(str(string)[:-2]), int(str(string)[-2:]))
    return (hours * 60. + minutes) // 60.


def day_length(sunrise, sunset):
    return (sunset - sunrise) / 60.


def normalise_columns(df, columns=None, exclude_columns=None):
    if not columns:
        columns = df.columns
    columns = list(columns)
    if exclude_columns:
        for ec in exclude_columns:
            columns.remove(ec)
    for column in columns:
        if column not in ('year', 'date', 'trap', 'wnvpresent', 'block'):
            try:
                df[column] = normalise_numeric(df[column])
            except TypeError as e:
                print(df[column])
                print('Error normalizing column:', column)
                print(str(e))
                exit()
    return df


def normalise_series(pd_series):
    return (pd_series - pd_series.min()) / (pd_series.max() - pd_series.min() + 1e-10)


def normalise_numeric(series):
    return normalise_series(pd.to_numeric(series))


def address_zipcode(address):
    match = re.findall('\d{5,}', address)
    if match:
        return match[0]
    else:
        return None


def write_csv(func):
    # todo
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        func().to_csv('output_from_{}.csv'.format(func.__name__))
        return result
    return wrapper


def verbose(message=None, on=False):
    # todo
    def wrapper(func):
        if on:
            print(message)
    return wrapper
