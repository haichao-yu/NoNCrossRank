import scipy.io as sio

import AUCEvaluation

# print sio.whosmat("../data/CRResults.mat")
data = sio.loadmat("../data/CRResults.mat")

ExpandSeeds = data['ExpandSeeds']
RankScoreRecord = data['RankScoreRecord']
RankRecord = data['RankRecord']
AllGeneID = data['AllGeneID']

AUCEvaluation.auc_evaluation(RankRecord, ExpandSeeds, AllGeneID)
