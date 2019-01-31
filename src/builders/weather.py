import numpy as np
import pandas as pd
from src.builders.utils import (
    write_csv, verbose, normalise_columns, year, month, week,
    get_df, str_to_date, string_time_to_minutes, day_length
)


COLS = [
    'date', 'tmax', 'tmin', 'tavg', 'depart', 'dewpoint',
    'heat', 'cool', 'sunrise', 'sunset', 'snowfall', 'preciptotal', 'stnpressure',
    'sealevel', 'resultspeed', 'resultdir', 'avgspeed'
]

def build_weather():
    df = get_df('weather')
    df = df[df.station == 1]
    df = prepare_weather(df)
    df = extra_features(df)
    return df


def build_aggregate_weather(by='year'):
    pass
    # todo
    # get time-windowed average features for weather.
    df = get_df('weather')
    df = df[df.station == 1]
    df = prepare_weather(df)
    df_agg = df.groupby(by).mean().reset_index()
    row = list(df_agg.values.flatten())
    df_row = pd.DataFrame([row], columns=['w' + str(i) for i in range(row.shape[0])])
    # todo: most columns are vanishing. thanks pandas
    # df_agg.rename(columns={c: c + '_w_agg' for c in df_agg.columns if c != by}, inplace=True)
    df_agg.to_csv('inspect.csv')
    return df_agg


def prepare_weather(df):
    df = df.replace('M', None, regex=True)
    df = df.replace('-', None, regex=True)
    df = df.replace('T', '0.00001', regex=True)
    df = df.fillna(method='pad')
    df['date'] = df['date'].apply(str_to_date)
    df['year'] = df['date'].apply(year)
    df['month'] = df['date'].apply(month)
    df['week'] = df['date'].apply(week)
    return df


def extra_features(df):
    df['sunrise'] = df['sunrise'].apply(string_time_to_minutes)
    df['sunset'] = df['sunset'].apply(string_time_to_minutes)
    df['daylight'] = df.apply(lambda row: day_length(row['sunrise'], row['sunset']), axis=1)
    df['dl_binary'] = (df['daylight'] > 10).astype(int)
    years = []
    for year, year_df in df.groupby('year'):
        year_df['rainprev'] = (np.hstack([np.array([0] * 2), year_df['preciptotal'][:-2].values]).astype(float) > 0).astype(int)
        year_df['dpprev'] = np.hstack([np.array([0] * 1), year_df['dewpoint'][:-1].values])
        year_df['wbprev'] = np.hstack([np.array([0] * 1), year_df['wetbulb'][:-1].values])
        year_df['cumtmax'] = rolling_average(year_df['tmax'], window=11)
        year_df['cumtmin'] = rolling_average(year_df['tmin'], window=11)
        year_df['cumdir'] = rolling_average(year_df['resultdir'], window=2)
        years.append(year_df)
    df = pd.concat(years)
    return df


def rolling_average(series, window, min_periods=1):
    return series.rolling(window=window, min_periods=min_periods).mean()


if __name__ == '__main__':
    build_aggregate_weather(by='month')
