import math
import numpy as np
import matplotlib.pyplot as plt


rng = np.random.default_rng()

def kmeans(data, k, initial_centroids):
    centroids = initial_centroids

    for n in range(100):
        new_clusters = [[] for _ in range(k)]
        for point in data:
            distances = [np.linalg.norm(point - centroid) for centroid in centroids]
            new_clusters[np.argmin(distances)].append(point)

        new_centroids = [np.mean(cluster, axis=0) for cluster in new_clusters]

        if np.array_equal(centroids, new_centroids):
            return new_centroids
        else:
            centroids = new_centroids
    return new_centroids
            
  
def dp_kmeans(data, k, Delta, epsilon, s, patience, initial_centroids):
    centroids = initial_centroids
    best_centroids = [[] for _ in range(k)]
    num_iterations_diverging = 0
    best_total_distances_to_centroids = math.inf
    # Be wary of floating point death
    while num_iterations_diverging < patience: 
        # Geometric privacy budget schedule
        epsilon /= s
        clusters = [[] for _ in range(k)]
        curr_distances_to_centroids = []
        for point in data:
            distances = [np.linalg.norm(point - centroid) for centroid in centroids]
            # Add sqrt(2)/epsilon laplace noise
            try:
                distances += rng.laplace(0, k * Delta / (epsilon * (s-1)), k)
            except ZeroDivisionError:
                break
            best_distance_idx = np.argmin(distances)
            clusters[best_distance_idx].append(point)
            curr_distances_to_centroids.append(distances[best_distance_idx])

        new_centroids = [np.mean(cluster, axis=0) for cluster in clusters]

        # Essentially the mean squared error
        total_curr_distances_to_centroids = np.linalg.norm(curr_distances_to_centroids)
        
        if total_curr_distances_to_centroids >= best_total_distances_to_centroids:
            num_iterations_diverging += 1
        else:
            best_total_distances_to_centroids = total_curr_distances_to_centroids
            best_centroids = new_centroids
            num_iterations_diverging = 0

        if np.array_equal(centroids, new_centroids):
            return best_centroids
        else:
            centroids = new_centroids

    return best_centroids

def accuracy(predicted_centroids, true_clusters, true_centroids):

    # This is horrible. Oh well.
    reordered_pred_centroids = [[] for _ in range(4)]
    for n in [3,2,1,0]:
        distances = [np.linalg.norm(true_centroids[n] - pred_centroid) for pred_centroid in predicted_centroids]
        idx = np.argmin(distances)
        reordered_pred_centroids[n] = predicted_centroids[idx]
        del predicted_centroids[idx]

    correct = 0
    for n in range(4):
        for point in true_clusters[n]:
            distances = [np.linalg.norm(point - centroid) for centroid in reordered_pred_centroids]
            if n == np.argmin(distances):
                correct += 1
    return correct / 400

def main():
    data = np.loadtxt("kmeansexample.asc")
    true_clusters = [data[0:100], data[100:200], data[200:300], data[300:400]]
    true_centroids = [np.mean(cluster, axis = 0) for cluster in true_clusters]
    epsilons = [0.5, 1, 2, 4, 8, 16]
    accuracies = []


    # 300 iterations takes ~3 minutes on a good laptop
    for _ in range(300):

        # For testing purposes, make sure each kmeans begins with equal initial centroids
        # I'm opting to initialize centroids using random data points,
        #   otherwise there's a chance that some clusters 
        #   get stuck with no points, and everything breaks
        idxs = np.random.choice(np.arange(len(data)), size=4, replace=False)
        initial_centroids = [data[idx] for idx in idxs] + rng.laplace(0, math.sqrt(2) / 8, (4,2))

        run_accuracies = []

        for eps in epsilons:
            # Slicing initial_centroids creates a shallow copy
            pred_centroids = dp_kmeans(data, 4, math.sqrt(2), eps, 2, 3, initial_centroids[:])
            run_accuracies.append(accuracy(pred_centroids, true_clusters, true_centroids))

        kmeans_centroids = kmeans(data, 4, initial_centroids[:])

        run_accuracies.append(accuracy(kmeans_centroids, true_clusters, true_centroids))
        accuracies.append(run_accuracies)
    
    # Average over all runs
    accuracies = np.mean(accuracies, axis=0)

    

    # Plotting stuff
    x_labels = [str(e) for e in epsilons] + ["No privacy"]
    x_vals = epsilons + [epsilons[-1]*4]
    fig, ax = plt.subplots()
    ax.plot(x_vals, accuracies, marker='o', linestyle='-')
    ax.set_xscale('log')

    ax.set_xticks(x_vals)
    ax.set_xticklabels(x_labels)

    ax.set_ylim(bottom=0, top=1)
    ax.set_xlabel('Epsilon')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Epsilon (Log Scale X-Axis)')

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

    



    
    
    return 0

if __name__ == "__main__":
    main()