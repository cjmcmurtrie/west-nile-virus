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


def day_to_sine(day):
    # see: https://math.stackexchange.com/questions/650223/formula-for-sine-wave-that-lines-up-with-calendar-seasons
    return math.sin(2 * math.pi / 365 * (day - 81.75))


def normalise(pd_series):
    return (pd_series - pd_series.min()) / (pd_series.max() + pd_series.min())


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

