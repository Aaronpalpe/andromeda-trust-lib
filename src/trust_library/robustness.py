import numpy as np

def compute_psi(expected, actual, bins=10):
    # crea contenedores de bins
    breakpoints = np.linspace(0, 1, bins+1)
    exp_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    act_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    psi_value = np.sum((exp_percents - act_percents) * np.log(exp_percents / act_percents))
    return psi_value

# uso
train_dist = np.random.rand(1000)
prod_dist  = np.random.rand(1000) * 1.1

psi_score = compute_psi(train_dist, prod_dist)
print("PSI:", psi_score)
