import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from keras.layers import Dense
from keras.models import Sequential

import config as cfg
import annealing

np.random.seed(cfg.RANDOM_SEED)


class ProcessedData(object):
    def __init__(self, x, y, target_names, feature_names):
        scaler = StandardScaler()

        self.x = scaler.fit_transform(x)  # Scaling x to 0, 1
        self.y = y
        self.target_names = target_names
        self.feature_names = feature_names
        self.random_seed = cfg.RANDOM_SEED

        self.features_dim = len(self.feature_names)
        self.target_dim = len(self.target_names)

    def split(self, test_size):
        return train_test_split(self.x, self.y, test_size=test_size, random_state=self.random_seed)


class NNModel(object):
    def __init__(self, name, input_dim, output_dim, batch_size=32, epochs=100):
        model = Sequential(name=name)
        for i in range(cfg.NUMBER_OF_LAYERS):
            model.add(Dense(cfg.NUMBER_OF_NODES_PER_LAYER, input_dim=input_dim, activation=cfg.ACTIVATION_FUNCTION))
        model.add(Dense(output_dim, activation='softmax'))

        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        model.summary()
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs

    def extract_weights(self):
        return [layer.get_weights() for layer in self.model.layers]

    def load_weights(self, weights):
        for layer, w in zip(self.model.layers, weights):
            layer.set_weights(w)

    def evaluate(self, x, y):
        return self.model.evaluate(x, y, batch_size=self.batch_size, verbose=0)

    def train(self, x, y):
        return self.model.fit(x, y, batch_size=self.batch_size, epochs=self.epochs, verbose=0)

    def predict(self, x):
        return self.model.predict(x, batch_size=self.batch_size)

    def save(self, filename):
        self.model.save(filename)


class SmartWeights(object):
    def __init__(self, weights_shapes, mu=0, sigma=1.0):
        self.weights_shapes = [[w.shape for w in w_layer] for w_layer in weights_shapes]
        self.weights_sigma = None
        self.weights_mus = None

        self.update_mus(mu, sigma)

    def generate(self):
        return [[np.random.normal(loc=weights_mus, scale=self.weights_sigma)
                 for weights_mus in layer_weights_mus]
                for layer_weights_mus in self.weights_mus]

    def update_mus(self, mu=0, sigma=1.0):
        self.weights_mus = [[np.random.normal(loc=mu, scale=sigma, size=s) for s in layer_s]
                            for layer_s in self.weights_shapes]
        self.weights_sigma = sigma


class NNAnnealing(annealing.Annealing):
    def __init__(self,
                 init_temp,
                 low_temp,
                 decay_factor,
                 number_of_iterations,
                 data: ProcessedData,
                 test_size=0.2):
        self.x_train, self.x_test, self.y_train, self.y_test = data.split(test_size=test_size)

        # Initialize model
        self.model = NNModel('annealed_model', data.features_dim, data.target_dim)

        # Initialize weights
        self.smart_weights = SmartWeights(self.model.extract_weights())

        super().__init__(init_temp,
                         low_temp,
                         decay_factor,
                         number_of_iterations,
                         self.smart_weights.generate())

    def random_state(self, state):
        return self.smart_weights.generate()

    def energy_function(self, weights):
        self.model.load_weights(weights)
        return self.model.evaluate(self.x_train, self.y_train)[0]

    def update(self, weights, energy):
        self.current_state = weights
        self.current_energy = energy
        self.smart_weights.update_mus()

    def evaluate(self, x, y):
        self.model.load_weights(self.best_state)
        return self.model.evaluate(x, y)


def load_data() -> ProcessedData:
    iris = load_iris()
    y = iris['target']

    encoder = OneHotEncoder()
    processed_y = encoder.fit_transform(y[:, np.newaxis]).toarray()
    return ProcessedData(
        x=iris['data'],
        y=processed_y,
        target_names=iris['target_names'],
        feature_names=iris['feature_names']
    )


def start_process():
    processed_data = load_data()

    annealing = NNAnnealing(
        data=processed_data,
        init_temp=30.0,
        low_temp=0.0000001,
        decay_factor=0.1,
        number_of_iterations=10000,
        test_size=0.5
    )
    annealing.start()
    x_eval, y_eval = annealing.x_test, annealing.y_test

    annealing_res = annealing.evaluate(x_eval, y_eval)

    annealing.model.save('annealing.h5')
    nn_model = NNModel('backprop_model', processed_data.features_dim, processed_data.target_dim)
    nn_model.train(annealing.x_train, annealing.y_train)
    nn_res = nn_model.evaluate(x_eval, y_eval)
    nn_model.save('backprop.h5')

    print(f'Annealing: {annealing_res}')
    print(f'NN: {nn_res}')


if __name__ == '__main__':
    start_process()
