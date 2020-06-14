#coding:UTF-8
import numpy as np
## 判断每一个特征是否存在奇异值点，若存在则将该特征删除，否则保留该特征
def MAD_outlier(x_train):
    n_feature = x_train.shape[1]
    n_sample = x_train.shape[0]
    median = np.median(x_train, axis=0)

    feature_left = []
    MAD = []
    for i in range(n_feature):
        fea = x_train[:, i]
        AD = []
        for j in range(n_sample):
            AD_temp = abs(fea[j] - median[i])
            AD.extend([AD_temp])
        MAD.extend([np.median(AD)])
        fea_outlier = np.array(AD > 3*MAD[i], dtype='bool')
        fea_outlier = np.nonzero(fea_outlier)[0]
        if len(fea_outlier) == 0:
            feature_left.extend([i])

    return feature_left

