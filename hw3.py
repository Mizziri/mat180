from collections import Counter
import math
import random

import matplotlib.pyplot as plt

random.seed(1729)

# q(D,c) = DATASET[c]
# Want to select a color c in R with probability ~ e^(epsilon/2 * DATASET[c]/Delta)
def prob(dataset, label, epsilon, sensitivity):
    return math.exp(0.5 * epsilon * dataset[label] / sensitivity)

# Accumulate probability masses, normalize, and randomly sample accordingly
def select_colors(dataset, labels, epsilon, sensitivity, num_simulations):
    prob_masses = [prob(dataset, label, epsilon, sensitivity) for label in labels]
    total_prob_mass = sum(prob_masses)
    normalized = [prob / total_prob_mass for prob in prob_masses]
    return Counter(random.choices(labels, weights = normalized, k = num_simulations))

# The following helper is ~50% AI-generated.
def make_histogram(counts, labels, epsilon, num_simulations):
    plt.figure(figsize=(8, 5))
    heights = [counts.get(color, 0) for color in labels]
    try:
        bars = plt.bar(labels, heights, color=labels)
    except:
        bars = plt.bar(labels, heights)

    for bar, count in zip(bars, heights):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight='bold'
        )
    plt.title(f"Exponential mechanism, {num_simulations} simulations, ε = {epsilon}")
    plt.xlabel("Color")
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    return 0


def main():
    # Output range / label set
    colors = ["Red", "Blue", "Green", "Yellow"]

    # Sensitivity is proven to be 1
    sensitivity = 1

    # Total number of responses is 80
    # Choosing 80000 means expected results for epsilon=0 are 1000x the data
    num_simulations = 80000

    ## Problem 2:

    # Privacy values
    epsilons = [0.1, 1]

    # Private responses, could maybe be a Counter for optimizations
    dataset = dict(zip(colors, [30, 25, 15, 10]))

    for eps in epsilons:
        samples = select_colors(dataset, colors, eps, sensitivity, num_simulations)
        make_histogram(samples, colors, eps, num_simulations)

    ## Problem 3:

    epsilon = 0.5

    dataset_2 = dict(zip(colors, [26, 24, 23, 27]))

    samples = select_colors(dataset_2, colors, epsilon, sensitivity, num_simulations)
    make_histogram(samples, colors, epsilon, num_simulations)

    # Extra: Distribution from problem 3 using epsilon = 0.1, 1
    for eps in epsilons:
        samples = select_colors(dataset_2, colors, eps, sensitivity, num_simulations)
        make_histogram(samples, colors, eps, num_simulations)


    return 0

if __name__ == "__main__":
    main()