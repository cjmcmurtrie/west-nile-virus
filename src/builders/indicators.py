import pandas as pd
from utils import get_df


COLS = {
    'zcta5': 'zipcode',
    'latitude': 'ind_lat',
    'longitude': 'ind_lon',
    'percent of housing crowded': 'crowding',
    'per capita income ': 'income pc',
    'total population': 'population',
    'not hispanic or latino, white alone': 'white',
    'not hispanic or latino, black or african american alone': 'afam',
    'not hispanic or latino, asian alone': 'asian',
    'hispanic or latino': 'hispanic',
    'total housing units': 'total housing',
    'vacant housing units': 'vacants',
}


def build_indicators():
    df = load_city_data()
    df.rename(columns=COLS, inplace=True)
    df = df[list(COLS.values())]
    return df


def load_city_data():
    indicators = get_df('se_indicators')
    census = get_df('census')
    city = pd.merge(indicators, census, left_on='community area number', right_on='geogkey')
    zipcodes = get_df('comarea_zipcode')
    df = pd.merge(city, zipcodes, left_on='community area number', right_on='chgoca')
    df = df.groupby('zcta5').mean().reset_index()
    return df


if __name__ == '__main__':
    build_indicators()
