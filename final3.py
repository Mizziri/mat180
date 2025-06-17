import numpy as np
import csv

# Reads in the data and generates confusion matrices for each group
def generate_confusion_matrices(filename):
    # In a real setting, use a defaultdict here. 
    out = {
        "African-American": np.array([
            #PN PP
            [0, 0], # AN
            [0, 0], # AP
        ]),
        "Caucasian": np.array([
            #PN PP
            [0, 0], # AN
            [0, 0], # AP
        ]),
    }
    with open(filename, "r") as csv_file:
        reader = csv.DictReader(csv_file)
        for line in reader:
            race, recidivism_pred = line["race"], int(line["decile_score"])
            # Ignore cases when prediction is 50/50 or data point is in a different group
            # Here's where defaultdict would be convenient
            if recidivism_pred != 5 and (race == "African-American" or race == "Caucasian"):
                # Conveniently index into the array and increment the count
                out[race][int(line["two_year_recid"]), int(recidivism_pred > 5)] += 1
    return out

# The following code is adapted from my HW4 submission.
# P(PP) = (TP + FP) / (TP + FP + TN + FN)
def pred_pos_rate(mat):
    num_pp = sum(mat[:,1])
    total = num_pp + sum(mat[:,0])
    return num_pp / total

# P(PP | AP) = TP / (TP + FN)
def sensitivity(mat):
    num_ap = sum(mat[1,:])
    return mat[1,1] / num_ap

# P(PP | AN) = TN / (TN + FP)
def specificity(mat):
    num_an = sum(mat[0,:])
    return mat[0,1] / num_an

# P (AP | PP) = TP / (TP + FP)
def precision(mat):
    num_pp = sum(mat[:,1])
    return mat[1,1] / num_pp

# 2 * precision * sens / (precision + sens), noting that sensitivity = recall
def f1_score(mat):
    sens = sensitivity(mat)
    prec = precision(mat)
    return 2 * prec * sens / (prec + sens)

def evaluate_model(group_0, group_1, verbose = False, ratios = True):
    # Per legal precedent
    delta = 0.8
    print(f"Allowed ratio between prediction rates is {delta}")
    ppr_0, ppr_1 = pred_pos_rate(group_0), pred_pos_rate(group_1)
    sens_0, sens_1 = sensitivity(group_0), sensitivity(group_1)
    spec_0, spec_1 = specificity(group_0), specificity(group_1)
    satisfies_pred_parity = (ppr_0 / ppr_1 >= delta) and (ppr_1 / ppr_0 >= delta)
    satisfies_eq_opp = (sens_0 / sens_1 >= delta) and (sens_1 / sens_0 >= delta)
    satisfies_eq_odds = (spec_0 / spec_1 >= delta) and (spec_1 / spec_0 >= delta) and satisfies_eq_opp

    print("Positive prediction rates are:")
    print(ppr_0)
    print(ppr_1)
    if ratios:
        print("Positive prediction ratios are:")
        print(ppr_0 / ppr_1)
        print(ppr_1 / ppr_0)
    if satisfies_pred_parity:
        print("Predictive parity IS satisfied!")
    else:
        print("Predictive parity is NOT satisfied!")
    print("Sensitivities are:")
    print(sens_0)
    print(sens_1)
    if ratios:
        print("Sensitivity ratios are:")
        print(sens_0 / sens_1)
        print(sens_1 / sens_0)
    print("Specificities are:")
    print(spec_0)
    print(spec_1)
    if ratios:
        print("Specificity ratios are:")
        print(spec_0 / spec_1)
        print(spec_1 / spec_0)
    if satisfies_eq_odds:
        print("Equalized odds ARE satisfied!")
    else:
        print("Equalized odds are NOT satisfied!")

    if verbose:
        prec_0, prec_1 = precision(group_0), precision(group_1)
        f1_0, f1_1 = f1_score(group_0), f1_score(group_1)
        print("Precisions are:")
        print(prec_0)
        print(prec_1)
        print("F1 scores are:")
        print(f1_0)
        print(f1_1)

def main():
    confusion_matrices = generate_confusion_matrices("compas-scores.csv")
    evaluate_model(confusion_matrices["African-American"], 
                   confusion_matrices["Caucasian"],
                   verbose = False, # Whether to display precision/recall/F1
                   ratios = False,  # Whether to display ratios of metrics
                   )

    return 0

if __name__ == "__main__":
    main()