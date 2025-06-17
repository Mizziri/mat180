import numpy as np

# Confusion matrices here
group_0 = np.array([
    #PN   PP
    [150, 20 ], # AN
    [30,  100], # AP
])
group_1 = np.array([
    #PN   PP
    [180, 40], # AN
    [20,  60], # AP
])

# bias = |P(PP|A=0) - P(PP|A=1)|
def bias(unprotected_group, protected_group):
    # P(PP|A=0)
    num_unprot_pp = sum(unprotected_group[:,1])
    total_unprot = num_unprot_pp + sum(unprotected_group[:,0])
    prob_0 = num_unprot_pp / total_unprot

    # P(PP|A=1)
    num_prot_pp = sum(protected_group[:,1])
    total_prot = num_prot_pp + sum(protected_group[:,0])
    prob_1 = num_prot_pp / total_prot
    
    return abs(prob_0 - prob_1)

# |P(PP|A=0 ^ AP) - P(PP|A=1 ^ AP)|
def eq_opp_diff(unprotected_group, protected_group):
    # P(PP|A=0 ^ AP)
    num_unprot_ap = sum(unprotected_group[1,:])
    prob_0 = unprotected_group[1,1] / num_unprot_ap

    # P(PP|A=1 ^ AP)
    num_prot_ap = sum(protected_group[1,:])
    prob_1 = protected_group[1,1] / num_prot_ap

    return abs(prob_0 - prob_1)

# 1/2 (|P(PP|A=0 ^ AN) - P(PP|A=1 ^ AN)| + eq_opp_diff)
# Notice that we can't add/subtract all probabilities before 1-norm, 
#   otherwise we get unintended cancellation
# Divide by 2 at the end to normalize value between 0 and 1

# Could alternatively use sup or 2-norm instead of adding both 1-norms.
#   No normalizing in sup norm, normalize by sqrt(2) in 2-norm.
def eq_odds_diff(unprotected_group, protected_group, mean):
    # P(PP|A=0 ^ AN)
    num_unprot_an = sum(unprotected_group[0,:])
    prob_0 = unprotected_group[0,1] / num_unprot_an

    # P(PP|A=1 ^ AN)
    num_prot_an = sum(protected_group[0,:])
    prob_1 = protected_group[0,1] / num_prot_an

    return mean(abs(prob_0 - prob_1), eq_opp_diff(unprotected_group, protected_group))

def arithmetic_mean(a, b):
    return 0.5 * (a + b)

def evaluate_model(unprotected_group, protected_group):
    print("Statistical Parity Difference:")
    print(bias(unprotected_group,protected_group))
    print("Equal Opportunity Difference:")
    print(eq_opp_diff(unprotected_group,protected_group))
    print("Equal Odds Difference:")
    print(eq_odds_diff(unprotected_group,protected_group, arithmetic_mean))

def main():
    evaluate_model(group_0, group_1)

    return 0

if __name__ == "__main__":
    main()