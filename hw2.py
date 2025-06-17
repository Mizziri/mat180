import numpy as np
import math
import json
import csv
from env import *

# Seed random number generator
rng = np.random.default_rng()

# Laplace mechanism is defined as:
# Delta = l1_sensitivity(f)
# M(x,f,eps) = f(x) + laplace_noise(Delta/eps) (as vectors)
# k is size of query return (?)
def laplace_noise(eps, Delta, k):
    return rng.laplace(0, Delta / eps, k) 

# M(x,f,eps) = f(x) + N(0,2ln(1.25/delta)Delta^2/eps^2)
# -> sigma = 2ln(1.25/delta)*Delta^2/eps^2
def gauss_noise(eps, Delta, delta, n):
    sigma = 2 * math.log(1.25/delta) * ((Delta / eps) ** 2)
    return rng.normal(0, sigma, n)

# Histogram queries:
# Addition or removal   -> l1_sensitivity = 1 
# Replacement           -> l1_sensitivity = 2
# --> magnitude of laplace noise independent of # of bins, k

## Lemma: Y ~ Lap(sigma)
# P[|Y| >= t * sigma] <= e^-t

## Laplace Error Theorem:
# P[sup|f(x)-y| > ln(k/delta)*(Delta/epsilon)] <= delta
# For histograms, probability that any bin has error >ln(k/delta)/eps is at most delta

def private_histogram(data, num_bins, eps, Delta):
    counts = np.zeros(num_bins)
    noisy_data = data + laplace_noise(eps, Delta, len(data))
    lowest = min(noisy_data)
    highest = max(noisy_data)

    # Precompute histogram cutoffs
    cutoffs = [round(lowest + i * (highest-lowest) / num_bins) for i in range(num_bins+1)]

    def get_bin(n):
        for i in range(len(cutoffs)):
            if n >= cutoffs[i] and n <= cutoffs[i+1]:
                return i

    # This should probably use a counter object
    for entry in data:
        counts[get_bin(entry)] += 1

    return { 'counts': counts, 'cutoffs': cutoffs }



def private_average_laplace(data, eps):
    # l1-sensitivity of average is 1/n
    return np.mean(data + laplace_noise(eps, 1/len(data), 1))

def private_average_gauss(data, eps_delta):
    # Absorb the sensitivity into our epsilon.
    return np.mean(data + gauss_noise(eps_delta[0], eps_delta[1], 1/len(data) ** 2, 1))


def get_error(actual_avg, noisy_avg):
    return abs(actual_avg - noisy_avg)/actual_avg


def main():
    filename = "adult-domain.json"
    with open(filename, 'r') as f:
        items = json.load(f)

    with open("adult.csv", 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for line in reader:
            headers = line
            break # lol
        
    dataset = np.genfromtxt('adult.csv', delimiter=',')[1:].astype(int)
    length = len(dataset[:])

    features = ['age', 'occupation']

    for feature in features:
        data = dataset[:,list(headers.keys()).index(feature)]
        actual_avg = np.mean(data)
        actual_stddev = np.std(data)
        print(actual_avg)
        worst_neighbor = (np.sum(data) + max(data) - min(data))/len(data)
        l1_sens = abs(worst_neighbor - actual_avg)

        zscores_lap, zscores_gauss = np.zeros(NUM_TRIALS), np.zeros(NUM_TRIALS)

        for n in range(NUM_TRIALS):
            # noisy data for each
            lap = private_average_laplace(data, AVG_EPSILON_LAPLACE[feature])
            gauss = private_average_gauss(data, AVG_EPSILON_DELTA_GAUSS[feature])

            # z-score for each
            # zscores_lap[n] = abs(actual_avg - lap)/actual_stddev
            # zscores_gauss[n] = abs(actual_avg - gauss)/actual_stddev

    features = ['age', 'workclass', 'education']
    for feature in features:
        data = dataset[:,list(headers.keys()).index(feature)]
        hist = private_histogram(data, items[feature], EPSILON_HIST_UNCORR, NOISE_HIST_UNCORR)
        print(hist['counts'])


        
        avg_lap_zscore = np.mean(zscores_lap)
        avg_gauss_zscore = np.mean(zscores_gauss)
        print(avg_lap_zscore)
        print(avg_gauss_zscore)




    ## Task 1: Average Age
    ages = dataset[:,list(headers.keys()).index('age')]
    occupations = dataset[:,list(headers.keys()).index('occupation')]

    actual_avg_age = np.mean(ages)
    actual_avg_occupation = np.mean(occupations)
    #print(actual_avg_age)
    ## From lecture, the sensitivity of average function is 1/n
    age_l1_sens = 1/length
    print(age_l1_sens)
    print(actual_avg_age+age_l1_sens)
    print(actual_avg_age-age_l1_sens)

    l1 sensitivity for average:
    worst case: smallest age replaced by largest age or vice-versa, or deletion of largest/smallest
    maxage = max(ages)
    minage = min(ages)
    total_age = np.sum(ages)
    worst_cases = [0]*6
    worst_cases[0] = (total_age + maxage)/(length + 1)
    worst_cases[1] = (total_age + minage)/(length + 1)
    worst_cases[2] = (total_age - maxage)/(length - 1)
    worst_cases[3] = (total_age - minage)/(length - 1)
    worst_cases[4] = (total_age - maxage + minage)/length
    worst_cases[5] = (total_age + maxage - minage)/length
    print(worst_cases)

    sensitivities = [abs(actual_avg_age - case) for case in worst_cases]
    print(sensitivities)
    avg_age_sens = max(sensitivities)

    i = 0
    laplace_age_avgs, gauss_age_avgs = np.zeros(NUM_TRIALS), np.zeros(NUM_TRIALS)
    laplace_occupation_avgs, gauss_occupation_avgs = np.zeros(NUM_TRIALS), np.zeros(NUM_TRIALS)
    while i < NUM_TRIALS:
        laplace_trial_age = ages + laplace_noise(EPSILON_AGE_LAPLACE, age_l1_sens, 1)
        laplace_age_avgs[i] = np.mean(laplace_trial_age)

        gauss_trial_age = ages + gauss_noise(EPSILON_AGE_GAUSS, DELTA_AGE, age_l1_sens ** 2, 1)
        gauss_age_avgs[i] = np.mean(gauss_trial_age)

        laplace_trial_occupation = occupations + laplace_noise(EPSILON_OCCUPATION_LAPLACE, age_l1_sens, 1)
        laplace_occupation_avgs[i] = np.mean(laplace_trial_occupation)

        gauss_trial_occupation = occupations + gauss_noise(EPSILON_OCCUPATION_GAUSS, DELTA_OCCUPATION, age_l1_sens ** 2, 1)
        gauss_occupation_avgs[i] = np.mean(gauss_trial_occupation)

        i += 1
    laplace_age_accuracies = [(actual_avg_age - avg)/actual_avg_age for avg in laplace_age_avgs]
    gauss_age_accuracies = [(actual_avg_age - avg)/actual_avg_age for avg in gauss_age_avgs]
    
    print(np.mean(laplace_age_accuracies))
    print(np.mean(gauss_age_accuracies))

    laplace_occupation_accuracies = [(actual_avg_occupation - avg)/actual_avg_occupation for avg in laplace_occupation_avgs]
    gauss_occupation_accuracies = [(actual_avg_occupation - avg)/actual_avg_occupation for avg in gauss_occupation_avgs]
    
    print(np.mean(laplace_occupation_accuracies))
    print(np.mean(gauss_occupation_accuracies))

    

main()