#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# Define hyperparameters for the ATE model
hyperparameters_ate = {
    'epochs': np.arange(16, 257, 1),  # Integer space
    'batch_size': np.arange(32, 513, 1),  # Integer space
    'gamma': np.linspace(0, 1, 100)  # Continuous space
}

def random_key_encoding(population_size, hyperparameters):
    population = []
    for _ in range(population_size):
        individual = []
        for param, space in hyperparameters.items():
            if isinstance(space, np.ndarray):
                individual.append(np.random.choice(space))
            else:
                individual.append(np.random.choice(range(len(space))))
        population.append(individual)
    return np.array(population)

def objective_function(individual):
    return np.random.rand()


# In[2]:


def modify_individual_using_learning(ind_i, ind_k, hyperparameters, F):
    ind_new = ind_i.copy()
    for j in range(len(ind_i)):
        phi = np.random.uniform(0, F)
        if objective_function(ind_i) < objective_function(ind_k):
            ind_new[j] = ind_i[j] + phi * (ind_k[j] - ind_i[j])
        else:
            ind_new[j] = ind_k[j] + phi * (ind_i[j] - ind_k[j])
        # Ensure the new individual remains within bounds
        param_name = list(hyperparameters.keys())[j]
        if isinstance(hyperparameters[param_name], np.ndarray):
            ind_new[j] = np.clip(ind_new[j], hyperparameters[param_name].min(), hyperparameters[param_name].max())
        else:
            ind_new[j] = min(max(int(round(ind_new[j])), 0), len(hyperparameters[param_name]) - 1)
    return ind_new


# In[3]:


def abc_algorithm(population, generations, hyperparameters, limit=10, F=0.5):
    fitness = np.array([objective_function(ind) for ind in population])
    trial_counter = np.zeros(len(population))

    for gen in range(generations):
        # Employed Bee Phase
        for i in range(len(population)):
            k = np.random.choice([idx for idx in range(len(population)) if idx != i])
            candidate = modify_individual_using_learning(population[i], population[k], hyperparameters, F)
            fit_candidate = objective_function(candidate)
            if fit_candidate < fitness[i]:
                population[i] = candidate
                fitness[i] = fit_candidate
                trial_counter[i] = 0
            else:
                trial_counter[i] += 1

        # Onlooker Bee Phase
        fitness_prob = (fitness.max() - fitness) / (fitness.max() - fitness.min() + 1e-9)
        fitness_prob /= fitness_prob.sum()
        for _ in range(len(population)):
            i = np.random.choice(len(population), p=fitness_prob)
            k = np.random.choice([idx for idx in range(len(population)) if idx != i])
            candidate = modify_individual_using_learning(population[i], population[k], hyperparameters, F)
            fit_candidate = objective_function(candidate)
            if fit_candidate < fitness[i]:
                population[i] = candidate
                fitness[i] = fit_candidate
                trial_counter[i] = 0
            else:
                trial_counter[i] += 1

        # Scout Bee Phase
        for i in range(len(population)):
            if trial_counter[i] >= limit:
                population[i] = random_key_encoding(1, hyperparameters)[0]
                fitness[i] = objective_function(population[i])
                trial_counter[i] = 0

        print(f"Generation {gen}: Best Fitness = {np.min(fitness)}")

    return population, fitness

# Parameters
population_size = 50
generations = 20


# Initialize population using random key encoding
population = random_key_encoding(population_size, hyperparameters_asc)

# Running the optimization
final_population, final_fitness = abc_algorithm(population, generations, hyperparameters_asc)


# In[ ]:




