import numpy as np


def auc_value(Rank, n):
    """
    Compute AUC value with n false positives
    """

    '''
    Parameter initialization
    '''
    loop = len(Rank)
    numerator = 0.0  # must be float

    TP = 0  # true positive
    FP = 0  # false positive

    # nonzero(): find the index of each nonzero elements (refer to find() in MatLab)
    AllTP = len(np.nonzero(Rank == 1))

    '''
    Calculation loop
    '''
    for i in range(loop):

        if Rank[i] == 1:
            TP += 1
        else:
            FP += 1
            numerator += TP

        if FP >= n:
            break

    '''
    denominator = sum(find(topn(j, :) == 0)) - NUM_Positive * (Num_Positive + 1) / 2
    '''
    denominator = FP * AllTP

    if numerator == 0:
        z = 0
    else:
        z = numerator / denominator

    return z
