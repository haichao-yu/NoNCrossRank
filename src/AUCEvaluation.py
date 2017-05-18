import numpy as np
import AUCValue


def auc_evaluation(RankRecord, ExpandSeeds, AllGeneID):
    """
    AUC value evaluation
    """

    ROCn = np.zeros((6, RankRecord.shape[1]))
    topn = np.zeros((len(RankRecord[0, 0]), RankRecord.shape[1]))

    for j in range(ExpandSeeds.shape[0]):

        for k in range(len(RankRecord[0, 0])):

            real_now = AllGeneID[RankRecord[0, j][k, 0] - 1]  # ID of gene at rank k

            if real_now == ExpandSeeds[j]:
                topn[k, j] = 1
            else:
                topn[k, j] = 0

        ROCn[0, j] = AUCValue.auc_value(topn[:, j], 50)
        ROCn[1, j] = AUCValue.auc_value(topn[:, j], 100)
        ROCn[2, j] = AUCValue.auc_value(topn[:, j], 300)
        ROCn[3, j] = AUCValue.auc_value(topn[:, j], 500)
        ROCn[4, j] = AUCValue.auc_value(topn[:, j], 700)
        ROCn[5, j] = AUCValue.auc_value(topn[:, j], 1000)

    avg_ROCn = np.average(ROCn, 1)
    print "AUC50: " + str(avg_ROCn[0])
    print "AUC100: " + str(avg_ROCn[1])
    print "AUC300: " + str(avg_ROCn[2])
    print "AUC500: " + str(avg_ROCn[3])
    print "AUC700: " + str(avg_ROCn[4])
    print "AUC1000: " + str(avg_ROCn[5])
