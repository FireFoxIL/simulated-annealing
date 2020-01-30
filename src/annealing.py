import numpy as np

from abc import ABC, abstractmethod


class Annealing(ABC):
    def __init__(self,
                 init_temp,
                 low_temp,
                 decay_factor,
                 number_of_iterations,
                 init_state):

        # Annealing parameters
        self.temp = init_temp
        self.low_temp = low_temp
        self.decay_factor = decay_factor
        self.number_of_iterations = number_of_iterations

        # Current State
        self.current_state = init_state
        self.current_energy = self.energy_function(init_state)

        # Best State
        self.best_state = self.current_state
        self.best_energy = self.current_energy

    @abstractmethod
    def random_state(self, state):
        pass

    @abstractmethod
    def energy_function(self, state):
        pass

    @abstractmethod
    def update(self, new_state, new_energy):
        pass

    def epoch_update(self):
        pass

    def boltzman_function(self, energy):
        power = -energy / self.temp
        return np.exp(power)

    def update_temp(self):
        self.temp = self.temp * self.decay_factor

    def move(self):
        for _ in range(self.number_of_iterations):
            new_state = self.random_state(self.current_state)
            new_energy = self.energy_function(new_state)

            if new_energy < self.best_energy:
                self.update(new_state, new_energy)
                self.best_state = new_state
                self.best_energy = new_energy
            else:
                alpha = self.boltzman_function(new_energy) / self.boltzman_function(self.current_energy)
                if np.random.uniform() <= alpha:
                    self.update(new_state, new_energy)

    def start(self):
        epoch_number = 0
        while self.temp >= self.low_temp:
            self.move()
            self.update_temp()
            self.epoch_update()
            print(f'Epoch [{epoch_number}] - Current Best Energy: {self.best_energy}')
            epoch_number += 1
