# WOA-SVR3 - Notum þennan til að prenta ut .
# WOA-SVR Corrected Code / Fyrir Data leakage var að æfa mig á test data-inu liklega
# B
# bætum svo við adaptive boundary og convergence rate plot.
# boundary adjustments as the algorith converges.
# Bætum hér við Elastic Net
# Elastic Net virkaði ekki þannig við forum i Adaptive boundary plús PSO+WOA hybrid.
# virkaði ekki betur förum i dynamic convergene parameters plus dynamic boundary scaling
# nema nuna ætlum við að bæta við mismunandi boundaries til að reyna að fiska lausnina.þ
import random
from math import sqrt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# List of filenames within the base directory
FILENAMES = [
    "Location4_daytime_week1.csv",
    "Location4_daytime_week2.csv",
    "Location4_daytime_week3.csv",
    "Location4_daytime_week4.csv",
    "Location4_nighttime_week1.csv",
    "Location4_nighttime_week2.csv",
    "Location4_nighttime_week3.csv",
    "Location4_nighttime_week4.csv",
]

# Parameters for testing
BOUNDS_LIST = [
    ([1e-2, 1e-4, 1e-4], [1e2, 1.0, 1e1]),
    ([1e-3, 1e-5, 1e-5], [1e3, 1.5, 1e2]),
    ([1e-1, 1e-3, 1e-3], [1e1, 0.5, 1.0]),
]

ITER_LIST = [10, 25, 30, 35, 40]
AGENTS_LIST = [5, 10, 15, 20]
OUTPUT_CSV = "Location4_Results.csv"


class WOA:
    """Whale Optimization Algorithm (WOA) class with adaptive boundaries"""

    def __init__(self, obj_func, lb, ub, dim, n_agents=20, max_iter=50, data_set=None):
        self.obj_func = obj_func
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.dim = dim
        self.n_agents = n_agents
        self.max_iter = max_iter
        self.data_set = data_set
        self.convergence_curve = []

    def optimize(self):
        """Finds best agent and best score using dynamic convergence update scaling"""
        agents = np.random.uniform(self.lb, self.ub, (self.n_agents, self.dim))
        best_agent = np.zeros((self.dim,))
        best_score = float("inf")

        for t in range(self.max_iter):
            for i in range(self.n_agents):
                score = self.obj_func(agents[i], self.data_set)
                if score < best_score:
                    best_score = score
                    best_agent = agents[i].copy()

            self.convergence_curve.append(best_score)

            boundary_scale = 1 - (t / self.max_iter)
            current_lb = self.lb * boundary_scale
            current_ub = self.ub * boundary_scale

            a = 2 - t * (2 / self.max_iter)
            for i in range(self.n_agents):
                r1, r2 = random.random(), random.random()
                a, c = 2 * a * r1 - a, 2 * r2
                b, l, p = 1, np.random.uniform(-1, 1), random.random()

                if p < 0.5:
                    if abs(a) < 1:
                        d = abs(c * best_agent - agents[i])
                        agents[i] = best_agent - a * d
                    else:
                        rand_agent = agents[np.random.randint(0, self.n_agents)]
                        d = abs(c * rand_agent - agents[i])
                        agents[i] = rand_agent - a * d
                else:
                    d = abs(best_agent - agents[i])
                    agents[i] = d * np.exp(b * l) * np.cos(2 * np.pi * l) + best_agent

                agents[i] = np.clip(agents[i], current_lb, current_ub)

        return best_agent, best_score


def objective_function(params, data_set):
    """Objective function for SVR with cross-validation"""
    c, epsilon, gamma = params
    x_train, y_train = data_set
    if c <= 0 or epsilon <= 0 or gamma <= 0:
        return float("inf")

    svr_model = SVR(C=c, epsilon=epsilon, gamma=gamma)
    scores = cross_val_score(
        svr_model, x_train, y_train, cv=5, scoring="neg_mean_squared_error"
    )
    mse = -scores.mean()
    return mse


def get_data_split(filename):
    """Parses most important feature factors. Most correlated with power features,
    raises FileNotFoundError if required files do not exist within current directory"""
    try:
        df = pd.read_csv(filename)
        x = df[["windspeed_100m", "windspeed_10m", "windgusts_10m"]]
        y = df["Power"]
    except FileNotFoundError as e:
        raise e
    return train_test_split(x, y, test_size=0.2, random_state=42)


def evaluate_woa_and_log(data_split, bounds_list, iter_list, agents_list, output_csv):
    """Function to evaluate and log results for each file"""
    all_results = []

    for filename, ds in zip(FILENAMES, data_split):
        x_train, x_test, y_train, y_test = ds

        for bounds in bounds_list:
            lb, ub = bounds
            for iterations in iter_list:
                for agents in agents_list:
                    print(
                        f"Testing File: {filename}, Bounds: {bounds}, Iterations: {iterations}, Agents: {agents}"
                    )
                    woa = WOA(
                        obj_func=objective_function,
                        lb=lb,
                        ub=ub,
                        dim=3,
                        n_agents=agents,
                        max_iter=iterations,
                        data_set=(x_train, y_train),
                    )
                    best_params, _ = woa.optimize()

                    c_best, epsilon_best, gamma_best = best_params
                    svr_best = SVR(C=c_best, epsilon=epsilon_best, gamma=gamma_best)
                    svr_best.fit(x_train, y_train)
                    y_train_pred = svr_best.predict(x_train)
                    y_test_pred = svr_best.predict(x_test)

                    train_rmse = sqrt(mean_squared_error(y_train, y_train_pred))
                    test_rmse = sqrt(mean_squared_error(y_test, y_test_pred))
                    test_r2 = r2_score(y_test, y_test_pred)

                    all_results.append(
                        {
                            "File": filename,
                            "Bounds": bounds,
                            "Iterations": iterations,
                            "Agents": agents,
                            "Best Params": best_params,
                            "Train RMSE": train_rmse,
                            "Test RMSE": test_rmse,
                            "Test R²": test_r2,
                        }
                    )

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)

    data_split = []
    for filename in FILENAMES:
        data_split.append(get_data_split(filename))

    # Run the evaluation and save the results
    evaluate_woa_and_log(data_split, BOUNDS_LIST, ITER_LIST, AGENTS_LIST, OUTPUT_CSV)
