#WOA-SVR3 - Notum þennan til að prenta ut .
# WOA-SVR Corrected Code / Fyrir Data leakage var að æfa mig á test data-inu liklega
# B
# bætum svo við adaptive boundary og convergence rate plot.
# boundary adjustments as the algorith converges.
# Bætum hér við Elastic Net
# Elastic Net virkaði ekki þannig við forum i Adaptive boundary plús PSO+WOA hybrid.
# virkaði ekki betur förum i dynamic convergene parameters plus dynamic boundary scaling
# nema nuna ætlum við að bæta við mismunandi boundaries til að reyna að fiska lausnina.þ
import os
import pandas as pd
import csv
import random
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Base directory for your data files
base_dir = r'C:\Users\Gunnar Ólafsson\OneDrive - Reykjavik University\Desktop\Location4_skipt_vikur'

# List of filenames within the base directory
filenames = [
    "Location4_daytime_week1.csv",
    "Location4_daytime_week2.csv",
    "Location4_daytime_week3.csv",
    "Location4_daytime_week4.csv",
    "Location4_nighttime_week1.csv",
    "Location4_nighttime_week2.csv",
    "Location4_nighttime_week3.csv",
    "Location4_nighttime_week4.csv"
]

# Whale Optimization Algorithm (WOA) class with adaptive boundaries
class WOA:
    def __init__(self, obj_func, lb, ub, dim, n_agents, max_iter):
        self.obj_func = obj_func
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.dim = dim
        self.n_agents = n_agents
        self.max_iter = max_iter
        self.convergence_curve = []

    def optimize(self):
        agents = np.random.uniform(self.lb, self.ub, (self.n_agents, self.dim))
        best_agent = np.zeros((self.dim,))
        best_score = float("inf")

        for t in range(self.max_iter):
            for i in range(self.n_agents):
                score = self.obj_func(agents[i])
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
                A, C = 2 * a * r1 - a, 2 * r2
                b, l, p = 1, np.random.uniform(-1, 1), random.random()

                if p < 0.5:
                    if abs(A) < 1:
                        D = abs(C * best_agent - agents[i])
                        agents[i] = best_agent - A * D
                    else:
                        rand_agent = agents[np.random.randint(0, self.n_agents)]
                        D = abs(C * rand_agent - agents[i])
                        agents[i] = rand_agent - A * D
                else:
                    D = abs(best_agent - agents[i])
                    agents[i] = D * np.exp(b * l) * np.cos(2 * np.pi * l) + best_agent

                agents[i] = np.clip(agents[i], current_lb, current_ub)

        return best_agent, best_score

# Objective function for SVR with cross-validation
def objective_function(params):
    C, epsilon, gamma = params
    if C <= 0 or epsilon <= 0 or gamma <= 0:
        return float("inf")
    
    svr_model = SVR(C=C, epsilon=epsilon, gamma=gamma)
    scores = cross_val_score(svr_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    mse = -scores.mean()
    return mse

# Function to evaluate and log results for each file
def evaluate_woa_and_log(base_dir, filenames, bounds_list, iter_list, agents_list, output_csv):
    all_results = []

    for filename in filenames:
        file_path = os.path.join(base_dir, filename)
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        df = pd.read_csv(file_path)
        X = df[['windspeed_100m', 'windspeed_10m', 'windgusts_10m']]
        y = df['Power']
        global X_train, X_test, y_train, y_test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        for bounds in bounds_list:
            lb, ub = bounds
            for iterations in iter_list:
                for agents in agents_list:
                    print(f"Testing File: {filename}, Bounds: {bounds}, Iterations: {iterations}, Agents: {agents}")
                    woa = WOA(obj_func=objective_function, lb=lb, ub=ub, dim=3, n_agents=agents, max_iter=iterations)
                    best_params, best_score = woa.optimize()

                    C_best, epsilon_best, gamma_best = best_params
                    svr_best = SVR(C=C_best, epsilon=epsilon_best, gamma=gamma_best)
                    svr_best.fit(X_train, y_train)
                    y_train_pred = svr_best.predict(X_train)
                    y_test_pred = svr_best.predict(X_test)

                    train_rmse = sqrt(mean_squared_error(y_train, y_train_pred))
                    test_rmse = sqrt(mean_squared_error(y_test, y_test_pred))
                    test_r2 = r2_score(y_test, y_test_pred)

                    all_results.append({
                        'File': filename,
                        'Bounds': bounds,
                        'Iterations': iterations,
                        'Agents': agents,
                        'Best Params': best_params,
                        'Train RMSE': train_rmse,
                        'Test RMSE': test_rmse,
                        'Test R²': test_r2
                    })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

# Parameters for testing
bounds_list = [
    ([1e-2, 1e-4, 1e-4], [1e2, 1.0, 1e1]),
    ([1e-3, 1e-5, 1e-5], [1e3, 1.5, 1e2]),
    ([1e-1, 1e-3, 1e-3], [1e1, 0.5, 1.0])
]
iter_list = [10, 25, 30, 35, 40]
agents_list = [5, 10, 15, 20]
output_csv = r'C:\Users\Gunnar Ólafsson\OneDrive - Reykjavik University\Desktop\Location4_Results.csv'

# Run the evaluation and save the results
evaluate_woa_and_log(base_dir, filenames, bounds_list, iter_list, agents_list, output_csv)