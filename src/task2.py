import pandas as pd
import numpy as np

from copy import deepcopy
import math

import annealing
import config as cfg
from animation import animate_history

np.random.seed(cfg.RANDOM_SEED)


class City(object):
    def __init__(self, index, name, geo_lat, geo_lon):
        self.index = index
        self.name = name
        self.geo_lat = geo_lat
        self.geo_lon = geo_lon

    def euc_distance(self, other):
        return math.sqrt((self.geo_lat - other.geo_lat) ** 2 + (self.geo_lon - other.geo_lon) ** 2)

    def distance(self, other):
        lat1, lon1 = self.geo_lat, self.geo_lon
        lat2, lon2 = other.geo_lat, other.geo_lon

        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)

        a = math.sin(dphi / 2) ** 2 + \
            math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2

        return 2 * cfg.R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def load_data(path) -> {City}:
    df = pd.read_csv(path)
    sorted_index = np.argsort(-df['population'].astype('int').values)[:cfg.NUMBER_OF_CITIES]
    selected_df = df[['city', 'geo_lat', 'geo_lon']].values[sorted_index]
    out = {}

    for i, (city, geo_lat, geo_lon) in enumerate(selected_df):
        out[i] = City(
            index=i,
            name=city,
            geo_lat=geo_lat,
            geo_lon=geo_lon
        )

    return out


class TravelersAnnealing(annealing.Annealing):
    def __init__(self,
                 init_temp,
                 low_temp,
                 decay_factor,
                 number_of_iterations,
                 country_map: {City}):
        self.country_map = country_map

        initial_state = [k for k in country_map.keys()]
        initial_state = np.random.permutation(initial_state)

        self.history = []

        super().__init__(init_temp,
                         low_temp,
                         decay_factor,
                         number_of_iterations,
                         initial_state)

    def energy_function(self, state):
        total_distance = 0
        for i in range(len(self.current_state)):
            first_city = self.country_map[state[i - 1]]
            second_city = self.country_map[state[i]]
            total_distance += first_city.distance(second_city)

        return total_distance

    def random_state(self, state):
        new_state = deepcopy(state)
        a = np.random.randint(0, len(state) - 1)
        b = np.random.randint(0, len(state) - 1)
        new_state[a] = state[b]
        new_state[b] = state[a]
        return new_state

    def update(self, new_state, new_energy):
        self.current_state = new_state
        self.current_energy = new_energy
        self.history.append(deepcopy(self.current_state))


if __name__ == '__main__':
    country_map = load_data('../city.csv')
    annealing = TravelersAnnealing(
        init_temp=25000,
        low_temp=10,
        decay_factor=0.6,
        number_of_iterations=200,
        country_map=country_map
    )

    annealing.start()

    animate_history(country_map, annealing.history)

