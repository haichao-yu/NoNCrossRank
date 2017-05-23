import os
import numpy as np
from scipy import sparse
import CR_Precomputation
import CR
import AUCEvaluation


def cr_cross_validation(alpha=0.5, c=0.85, MaxIter=1000, epsilon=1e-6):
    """
    Cross Rank leave-one-out cross validation on tissue-specific PPI networks.

    If no input parameters are provided, the default values will be used.

    :param alpha: a regularization parameter for cross-network consistency
    :param c: a regularization parameter for query preference
    :param MaxIter: the maximal number of iteration for updating ranking vector
    :param epsilon: a convergence parameter
    """

    '''
    Load NoN data
    '''
    data = np.load("../data/P_G_NoN.npy").item()
    PhenotypeSimNet = data['PhenotypeSimNet']
    PhenotypeID = data['PhenotypeID']
    AllGeneID = data['AllGeneID']
    TSGeneNets = data['TSGeneNets']
    TSGeneNetsID = data['TSGeneNetsID']
    Seeds = data['Seeds']
    TissueDict = data['TissueDict']

    '''
    Rename networks
    '''
    g = TSGeneNets.shape[1]  # the number of domain-specific networks
    G = PhenotypeSimNet  # the main network
    A = TSGeneNets  # the domain-specific networks
    A_ID = TSGeneNetsID  # the IDs of nodes in domain-specific networks
    A_Seeds = Seeds  # the seed/query nodes in domain-specific networks

    '''
    Cross Rank precomputation, this step only needs to be done once for a dataset
    '''
    PrecompFileName = 'CR_Precomp_Values_TPPI.npy'

    if os.path.isfile(PrecompFileName):
        print("A precomputation file has been detected ...")
    else:
        print("CR precomputation starts ...")
        CR_Precomputation.cr_precomputation(A, A_ID, G, PrecompFileName)

    print("Load the precomputation file ...")
    data = np.load(PrecompFileName).item()
    I_n = data['I_n']
    Anorm = data['Anorm']
    Ynorm = data['Ynorm']

    '''
    Leave-one-out cross validation
    '''
    # expand test genes s.t. test gene one by one
    ExpandSeeds = np.vstack(A_Seeds[0, :])

    # leave-one-out cross validation loop
    RankScoreRecord = np.zeros((1, ExpandSeeds.shape[0]), dtype=object)  # cell matrix
    RankRecord = np.zeros((1, ExpandSeeds.shape[0]), dtype=object)  # cell matrix
    TotalCounter = 0

    print "Leave-one-out cross validation starts ..."

    for j in range(g):

        for t in range(A_Seeds[0, j].shape[0]):

            # initialize query vector
            e = np.array([])

            # construct the aggregated query vector e = (e1, ..., eg)
            for i in range(g):

                if i == j:
                    head = len(e)  # Record head position in e/r for final evaluation

                tmp_e = np.zeros(A_ID[0, i].shape[0])

                if np.in1d(A_Seeds[0, j][t], A_Seeds[0, i]):
                    subn = A_Seeds[0, i].shape[0] - 1;
                    if subn != 0:
                        Fia = np.in1d(A_Seeds[0, i], A_ID[0, i])
                        seedidx = np.in1d(A_ID[0, i], A_Seeds[0, i]).nonzero()[0]
                        tmp_e[seedidx] = 1.0 / subn
                        Fia1 = np.in1d(A_Seeds[0, j][t], A_ID[0, i])
                        seedidx1 = np.in1d(A_ID[0, i], A_Seeds[0, j][t]).nonzero()[0]
                        tmp_e[seedidx1] = 0  # leave one out (A_Seeds[0, j][t])
                else:
                    subn = A_Seeds[0, i].shape[0]
                    Fia = np.in1d(A_Seeds[0, i], A_ID[0, i])
                    seedidx = np.in1d(A_ID[0, i], A_Seeds[0, i]).nonzero()[0]
                    tmp_e[seedidx] = 1.0 / subn

                e = np.hstack((e, tmp_e))

                if i == j:
                    tail = e.size

            # CR
            e = e.reshape(len(e), 1)
            e = sparse.csc_matrix(e)
            [r, Objs, Deltas] = CR.cr(Anorm, Ynorm, I_n, e, alpha, c, MaxIter, epsilon)

            # record results
            RankScore = r[head:tail, 0]
            # RankScore = (round(RankScore * 1e16)) / 1e16
            FullRankScore = np.zeros((AllGeneID.shape[0], 1))
            Fia = np.in1d(A_ID[0, j], AllGeneID[:, 0])
            idx = np.in1d(AllGeneID[:, 0], A_ID[0, j]).nonzero()[0]
            RankScore = RankScore.todense().getA1()
            FullRankScore[idx, 0] = RankScore

            SeedGeneList = np.setdiff1d(A_Seeds[0, j], A_Seeds[0, j][t])
            proj = np.intersect1d(SeedGeneList, AllGeneID[:, 0])
            ia = np.in1d(SeedGeneList, proj).nonzero()[0]
            idx = np.in1d(AllGeneID[:, 0], proj).nonzero()[0]
            FullRankScore[idx, 0] = 0

            RankScoreRecord[0, TotalCounter] = FullRankScore

            # sort results
            SortFullRankScore = np.flip(np.sort(FullRankScore, axis=0), axis=0)
            IX = np.flip(np.argsort(FullRankScore, axis=0), axis=0)

            RankRecord[0, TotalCounter] = IX

            print("Finished Number of Folds/Total Number of Folds: " + str(TotalCounter) + "/" + str(ExpandSeeds.shape[0]))

            TotalCounter += 1

    '''
    Save results
    '''
    print("Leave-one-out cross validation finishes, save results ...")

    resultFileName = 'CRResults.npy'

    results = {}
    results['RankScoreRecord'] = RankScoreRecord
    results['RankRecord'] = RankRecord
    results['ExpandSeeds'] = ExpandSeeds
    results['AllGeneID'] = AllGeneID

    np.save(resultFileName, results)

    '''
    Evaluation
    '''
    print("AUC value evaluation starts ...")
    AUCEvaluation.auc_evaluation(RankRecord, ExpandSeeds, AllGeneID)
