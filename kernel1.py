'''
Udates
http://stackoverflow.com/questions/10565868/what-is-the-criteria-for-choosing-number-of-hidden-layers-and-nodes-in-hidden-la

Rule of thumb for # of input layers
input layer: the size of my data vactor (the number of features in my model) + 1
for the bias node and not including the response variable, of course
output layer: soley determined by my model: regression (one node) versus classification
(number of nodes equivalent to the number of classes, assuming softmax)
hidden layer: to start, one hidden layer with a number of nodes equal to the size of the input layer.
The "ideal" size is more likely to be smaller (i.e, some number of nodes between the number in the input layer and the number in the output layer) rather than larger--again, this is just an empirical observation, and the bulk of this observation is my own experience. If the project justified the additional time required, then i start with a single hidden layer comprised of a small number of nodes, then (as i explained just above) i add nodes to the Hidden Layer, one at a time, while calculating the generalization error, training error, bias, and variance. When generalization error has dipped and just before it begins to increase again, the number of nodes at that point is my choice. See figure below.
'''
import numpy as np
import pandas as pd
import datetime
import csv
from sklearn import metrics
from sklearn.utils import shuffle

species_map = {'CULEX RESTUANS': "100000",
               'CULEX TERRITANS': "010000",
               'CULEX PIPIENS': "001000",
               'CULEX PIPIENS/RESTUANS': "101000",
               'CULEX ERRATICUS': "000100",
               'CULEX SALINARIUS': "000010",
               'CULEX TARSALIS': "000001",
               'UNSPECIFIED CULEX': "001100"}  # hack!


def date(text):
    return datetime.datetime.strptime(text, "%Y-%m-%d").date()


def precip(text):
    TRACE = 1e-3
    text = text.strip()
    if text == "M":
        return None
    if text == "-":
        return None
    if text == "T":
        return TRACE
    return float(text)


def ll(text):
    return int(float(text) * 100) / 100


def impute_missing_weather_station_values(weather):
    # Stupid simple
    for k, v in weather.items():
        if v[0] is None:
            v[0] = v[1]
        elif v[1] is None:
            v[1] = v[0]
        for k1 in v[0]:
            if v[0][k1] is None:
                v[0][k1] = v[1][k1]
        for k1 in v[1]:
            if v[1][k1] is None:
                v[1][k1] = v[0][k1]


def load_weather():
    weather = {}
    for line in csv.DictReader(open("data/weather.csv")):
        for name, converter in {"Date": date,
                                "Tmax": float, "Tmin": float, "Tavg": float,
                                "DewPoint": float, "WetBulb": float,
                                "PrecipTotal": precip, "Sunrise": precip, "Sunset": precip,
                                "Depart": float, "Heat": precip, "Cool": precip,
                                "ResultSpeed": float, "ResultDir": float, "AvgSpeed": float,
                                "StnPressure": float, "SeaLevel": float}.items():
            x = line[name].strip()
            line[name] = converter(x) if (x != "M") else None
        station = int(line["Station"]) - 1
        assert station in [0, 1]
        dt = line["Date"]
        if dt not in weather:
            weather[dt] = [None, None]
        assert weather[dt][station] is None, "duplicate weather reading {0}:{1}".format(dt, station)
        weather[dt][station] = line
    impute_missing_weather_station_values(weather)
    return weather


def load_training():
    training = []
    for line in csv.DictReader(open("data/train.csv")):
        for name, converter in {"Date": date,
                                "Latitude": ll, "Longitude": ll,
                                "NumMosquitos": int, "WnvPresent": int}.items():
            line[name] = converter(line[name])
        training.append(line)
    return training


def load_testing():
    training = []
    for line in csv.DictReader(open("data/test.csv")):
        for name, converter in {"Date": date,
                                "Latitude": ll, "Longitude": ll}.items():
            line[name] = converter(line[name])
        training.append(line)
    return training


def closest_station(lat, longi):
    # Chicago is small enough that we can treat coordinates as rectangular.
    stations = np.array([[41.995, -87.933],
                         [41.786, -87.752]])
    loc = np.array([lat, longi])
    deltas = stations - loc[None, :]
    dist2 = (deltas ** 2).sum(1)
    return np.argmin(dist2)


def normalize(X, mean=None, std=None):
    count = X.shape[1]
    if mean is None:
        mean = np.nanmean(X, axis=0)
    for i in range(count):
        X[np.isnan(X[:, i]), i] = mean[i]
    if std is None:
        std = np.std(X, axis=0)
    for i in range(count):
        X[:, i] = (X[:, i] - mean[i]) / std[i]
    return mean, std


def scaled_count(record):
    SCALE = 9.0
    if "NumMosquitos" not in record:
        # This is test data
        return 1
    return int(np.ceil(record["NumMosquitos"] / SCALE))


def assemble_X(base, weather):
    X = []
    for b in base:
        date = b["Date"]
        date2 = np.sin((2 * 3.1459265359 * date.day) / 365 * 24)
        date3 = np.cos((2 * 3.1459265359 * date.day) / 365 * 24)
        date4 = np.sin((2 * 3.1459265359 * date.month) / 365)
        lat, longi = b["Latitude"], b["Longitude"]
        case = [date.year, date.month, date4, date.day, date.weekday(), date2, date3, lat, longi]
        # Look at a selection of past weather values
        for days_ago in [1, 2, 3, 5, 8, 12]:
            day = date - datetime.timedelta(days=days_ago)
            for obs in ["Tmax", "Tmin", "Tavg", "DewPoint", "WetBulb", "PrecipTotal", "Depart", "Sunrise", "Sunset",
                        "Heat", "Cool", "ResultSpeed", "ResultDir"]:
                station = closest_station(lat, longi)
                case.append(weather[day][station][obs])
                # case.append(np.sin(2*3.1459265359*weather[day][station][obs]))
        # Specify which mosquitos are present
        species_vector = [float(x) for x in species_map[b["Species"]]]
        case.extend(species_vector)
        # Weight each observation by the number of mosquitos seen. Test data
        # Doesn't have this column, so in that case use 1. This accidentally
        # Takes into account multiple entries that result from >50 mosquitos
        # on one day.
        for repeat in range(scaled_count(b)):
            X.append(case)
    X = np.asarray(X, dtype=np.float32)
    return X


def assemble_y(base):
    y = []
    for b in base:
        present = b["WnvPresent"]
        for repeat in range(scaled_count(b)):
            y.append(present)
    return np.asarray(y, dtype=np.int32).reshape(-1, 1)


class AdjustVariable(object):
    def __init__(self, variable, target, half_life=20):
        self.variable = variable
        self.target = target
        self.half_life = half_life

    def __call__(self, nn, train_history):
        delta = self.variable.get_value() - self.target
        delta /= 2 ** (1.0 / self.half_life)
        self.variable.set_value(np.float32(self.target + delta))


def train():
    weather = load_weather()
    training = load_training()

    X = assemble_X(training, weather)
    mean, std = normalize(X)
    y = assemble_y(training)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    net = LogisticRegression(class_weight='balanced')
    # net = RandomForestClassifier()

    X, y = shuffle(X, y, random_state=123)
    net.fit(X, y)

    ix = int(X.shape[0] * 0.97)
    Xtr = X[:ix]
    Xte = X[ix:]
    ytr = y[:ix]
    yte = y[ix:]

    net.fit(Xtr, ytr)
    probas = net.predict_proba(Xte)[:, 1]
    print("ROC score", metrics.roc_auc_score(yte, probas))

    return net, mean, std


def submit(net, mean, std):
    weather = load_weather()
    testing = load_testing()
    X = assemble_X(testing, weather)
    normalize(X, mean, std)
    probas = net.predict_proba(X)[:, 1]
    ids = range(1, probas.shape[0] + 1)
    submission = pd.DataFrame({
        'Id': ids,
        'WnvPresent': probas
    })
    submission.to_csv('submissions/submission_{}.csv'.format(datetime.datetime.now()), index=False)


if __name__ == "__main__":
    net, mean, std = train()
    submit(net, mean, std)