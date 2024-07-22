import numpy as np

time_vars = np.loadtxt("time.txt")
space_var = np.loadtxt("space.txt")

seed = 0
rng = np.random.default_rng(seed)

space_variance = 0.1 * space_var
space_noise = rng.normal(0, np.sqrt(space_variance))
space_data = space_var + space_noise
np.savez("space_data", data=space_data, variance=space_variance)

time_data = time_vars[:, 1]
time_sol = time_vars[:, 2]
time_data_diff = time_data - time_sol
time_variance = np.mean(np.square(time_data_diff)) * np.ones(time_data.size)
np.savez("time_data", data=time_data, variance=time_variance)
