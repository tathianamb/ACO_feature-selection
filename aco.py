import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureSelectionACO(BaseEstimator, TransformerMixin):
    def __init__(self, clf, num_features, num_ants, num_iterations):
        self.clf = clf
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.num_features = num_features
        self.pheromone = np.ones(self.num_features)
        self.best_accuracy = 0.0
        self.best_selected_features = []
        self.iteration_values = []
        self.accuracy_values = []

    def fitness_function(self, features):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        if not features:
            return 0.0
        self.clf.fit(X_train[:, features], np.ravel(y_train))
        y_pred = self.clf.predict(X_test[:, features])
        return accuracy_score(y_test, y_pred)

    def generate_solution(self):
        solution = []
        for feature in range(self.num_features):
            prob = self.pheromone[feature] / np.sum(self.pheromone)
            if np.random.rand() < prob:
                solution.append(feature)
        return solution

    def update_pheromone(self, fitness):
        self.pheromone += fitness

    def update_best_solution(self, fitness, solution):
        if fitness > self.best_accuracy:
            self.best_accuracy = fitness
            self.best_selected_features = solution

    def fit(self, X, y, **fit_params):
        self.X = X
        self.y = y
        for iteration in range(self.num_iterations):
            for ant in range(self.num_ants):
                solution = self.generate_solution()
                fitness = self.fitness_function(solution)
                self.update_pheromone(fitness)
                self.update_best_solution(fitness, solution)

            self.iteration_values.append(iteration)
            self.accuracy_values.append(self.best_accuracy)

        return self

    def transform(self, X):
        return X[:, self.best_selected_features]
