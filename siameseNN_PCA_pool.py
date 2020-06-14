#-*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import xlrd
import xlwt
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
import concurrent.futures
from multiprocessing import Pool
from scipy import stats 
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import tensorflow as tf
from itertools import combinations, permutations
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA, KernelPCA
import time
from MAD_outlier import MAD_outlier
import os

## 读取数据   变化BMI
filefullpath = "DATA_Pre_Post6_Pairs.xlsx"
ycolume = "O"                 # 使用WeightLoss%来衡量减重效果好坏
df = pd.read_excel(filefullpath, sheet_name=2, skiprows=[0,1], header=None, usecols=ycolume)
ylabel = df.values
ylabel = ylabel.reshape(np.shape(ylabel)[0])

## 读取影像数据 相关系数特征 246模板 30135
filefullpath = "FCsZ_246ROI.xlsx"
xcolume = "A:AK"
df = pd.read_excel(filefullpath, sheet_name=0, skiprows=None, header=None, usecols=xcolume)
data = (df.values).T

ylabel_temp = list(ylabel)
idx1 = [i for i, x in enumerate(ylabel_temp) if x==1]
idx0 = [i for i, x in enumerate(ylabel_temp) if x==0]
data_x0 = data[idx0, :]
data_x1 = data[idx1, :]
feature_left0 = MAD_outlier(data_x0)
feature_left1 = MAD_outlier(data_x1)
feature_left = [x for x in feature_left0 if x in feature_left1]
# feature_left = MAD_outlier(data)
data = data[:, feature_left]
n_feature = data.shape[1]

## 参数
iterations = 10000
batch_size = 32
num_labels = 2
num_features = 36

tf.reset_default_graph()
def randomize_tensor(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:]
    shuffled_labels = labels[permutation,:]
    return shuffled_dataset, shuffled_labels

def siamese_loss(out1,out2,y,Q=5):
    Q = tf.constant(Q, name="Q",dtype=tf.float32)
    E_w = tf.sqrt(tf.reduce_sum(tf.square(out1-out2),1))   
    pos = tf.multiply(tf.multiply(1-y,2/Q),tf.square(E_w))
    neg = tf.multiply(tf.multiply(y,2*Q),tf.exp(-2.77/Q*E_w))                
    loss = pos + neg                 
    loss = tf.reduce_mean(loss)              
    return loss

def siamese(inputs):
    with tf.name_scope('fc1') as scope:
        w_fc1=tf.Variable(tf.truncated_normal(shape=[num_features,16],stddev=0.01,mean=0),name='w_fc1')
        b_fc1 = tf.Variable(tf.zeros(16),name='b_fc1')
        fc1 = tf.add(tf.matmul(inputs, w_fc1), b_fc1)
    with tf.name_scope('tanh_fc1') as scope:
        tanh_fc1 = tf.nn.tanh(fc1,name='tanh_fc1')

    with tf.name_scope('fc2') as scope:
        w_fc2=tf.Variable(tf.truncated_normal(shape=[16,8],stddev=0.01,mean=0),name='w_fc2')
        b_fc2 = tf.Variable(tf.zeros(8),name='b_fc2')
        fc2 = tf.add(tf.matmul(tanh_fc1,w_fc2),b_fc2)
    with tf.name_scope('tanh_fc2') as scope:
        tanh_fc2 = tf.nn.tanh(fc2,name='tanh_fc2')

    with tf.name_scope('fc3') as scope:
        w_fc3=tf.Variable(tf.truncated_normal(shape=[8,2],stddev=0.01,mean=0),name='w_fc3')
        b_fc3 = tf.Variable(tf.zeros(2),name='b_fc3')
        fc3 = tf.add(tf.matmul(tanh_fc2,w_fc3),b_fc3)
    
    return fc3

with tf.variable_scope('input_x1') as scope:
    x1 = tf.placeholder(tf.float32, shape=[None, num_features])
with tf.variable_scope('input_x2') as scope:
    x2 = tf.placeholder(tf.float32, shape=[None, num_features])
with tf.variable_scope('y') as scope:
    y = tf.placeholder(tf.float32, shape=[batch_size])

with tf.variable_scope('siamese') as scope:
    out1 = siamese(x1)
    scope.reuse_variables()
    out2 = siamese(x2)
with tf.variable_scope('metrics') as scope:
    global_step = tf.placeholder(tf.float32)
    decayed_learning_rate = tf.train.exponential_decay(0.0001, global_step, 500, 0.9, staircase=True)    
    loss = siamese_loss(out1, out2, y)
    optimizer = tf.train.AdamOptimizer(decayed_learning_rate).minimize(loss)

loss_summary = tf.summary.scalar('loss',loss)
merged_summary = tf.summary.merge_all()

label_pred_all = []
label_true_all = []
pred_prob = []
## 留一交叉验证
# loo = LeaveOneOut()
# loop = 0
def mainLoop(train_index, test_index, data, ylabel, loop):
# for train_index, test_index in loo.split(data):
    x_train, x_test = data[train_index], data[test_index]
    y_train, y_test = ylabel[train_index], ylabel[test_index]

    FCsZ_selected = []    # 选择的特征边编号(0-30134)
    FCsZ_selected = feature_left

    ## 标准化
    xMean = np.mean(x_train, axis=0)
    xStd = np.std(x_train, axis=0)
    x_train = (x_train - xMean) / xStd
    x_test = (x_test - xMean) / xStd

    ## ttest
    n_feature = x_train.shape[1]
    label_temp = list(y_train)
    idx0 = [i for i, x in enumerate(label_temp) if x == 0]
    idx1 = [i for i, x in enumerate(label_temp) if x == 1]
    x_0 = x_train[idx0, :]
    x_1 = x_train[idx1, :]
    [corr, corr_p] = stats.ttest_ind(x_0, x_1)
    number = (np.arange(0, n_feature)).reshape(1, n_feature)
    corr = (np.array(corr)).reshape(1, n_feature)
    corr_p = (np.array(corr_p)).reshape(1, n_feature)
    order = np.concatenate((number, corr, corr_p), axis=0)
    order_temp = order[:, order[2].argsort()]
    
    ROI_select = np.array(order_temp[2, :] < 0.05, dtype='bool')
    ROI_select = np.nonzero(ROI_select)[0]
    ROI_select_filter = list(order_temp[0, ROI_select])
    ROI_select_filter = [int(x) for x in ROI_select_filter]
    
    FCsZ_selected = [FCsZ_selected[i] for i in ROI_select_filter]

    ## 更新数据
    x_train = x_train[:, ROI_select_filter]
    x_test = x_test[:, ROI_select_filter]
    ## PCA
    pca = PCA(n_components=num_features, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None)
    pca.fit(x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)

    ## 排列组合 构造训练集
    pos_index = np.array(y_train == 1, dtype='bool')
    pos_index = np.nonzero(pos_index)[0]
    neg_index = np.array(y_train == 0, dtype='bool')
    neg_index = np.nonzero(neg_index)[0]

    pos_samples = x_train[pos_index, :]
    neg_samples = x_train[neg_index, :]

    num_pos_samples = pos_samples.shape[0]
    num_neg_samples = neg_samples.shape[0]

    sim_index_pos = np.array(list(combinations(np.arange(num_pos_samples), 2)))
    sim_index_neg = np.array(list(combinations(np.arange(num_neg_samples), 2)))

    sim_pos_data1 = pos_samples[sim_index_pos[:, 0], :]
    sim_pos_data2 = pos_samples[sim_index_pos[:, 1], :]
    sim_pos_data = np.concatenate((sim_pos_data1, sim_pos_data2), axis=1)
    sim_neg_data1 = neg_samples[sim_index_neg[:, 0], :]
    sim_neg_data2 = neg_samples[sim_index_neg[:, 1], :]
    sim_neg_data = np.concatenate((sim_neg_data1, sim_neg_data2), axis=1)
    sim_data = np.concatenate((sim_pos_data, sim_neg_data), axis=0)

    diff_data = np.zeros([1, (pos_samples.shape[1])*2])
    for i in range(num_pos_samples):
        for j in range(num_neg_samples):
            diff_data_temp = np.concatenate((pos_samples[i, :], neg_samples[j, :]), axis=0)
            diff_data_temp = diff_data_temp.reshape([1, diff_data_temp.shape[0]])
            diff_data = np.concatenate((diff_data, diff_data_temp), axis=0)
    diff_data = diff_data[1:diff_data.shape[0], :]

    sim_label = np.zeros(sim_data.shape[0])
    sim_label[:] = 0
    diff_label = np.zeros(diff_data.shape[0])
    diff_label[:] = 1

    data_train_all = np.concatenate((sim_data, diff_data), axis=0)
    label_train_all = np.concatenate((sim_label, diff_label), axis=0)
    label_train_all = (np.arange(num_labels) == label_train_all[:,None]).astype(np.float32)  # 相同为[1, 0], 不同为[0, 1], 取第二列

    with tf.Session() as sess:

        writer = tf.summary.FileWriter('./graph/siamese',sess.graph)
        sess.run(tf.global_variables_initializer())

        train_data, train_labels = randomize_tensor(data_train_all, label_train_all)
        for itera in range(iterations):
            offset = (itera * batch_size) % (train_labels.shape[0] - batch_size)          # (4*batch_size)%
            if (offset == 0 ):
                    train_data, train_labels = randomize_tensor(train_data, train_labels)   # 随机化一次
                    #create batch
            batch_data = train_data[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]      # 每一批数据4个

            xs_1 = batch_data[:, 0:num_features]
            xs_2 = batch_data[:, num_features:batch_data.shape[1]]
            y_s = batch_labels[:, 1]
            _,train_loss,summ = sess.run([optimizer,loss,merged_summary],feed_dict={x1:xs_1, x2:xs_2, y:y_s, global_step: itera})

            writer.add_summary(summ,itera)
            # if itera % 1000 == 1 :
            #     print('iter {},train loss {}'.format(itera, train_loss))
        # loop = loop + 1   # 交叉验证循环次数
        ## 测试
        test_out = sess.run(out1, feed_dict={x1:x_test})

        train_samples = np.concatenate((pos_samples, neg_samples), axis=0)
        train_out = sess.run(out1, feed_dict={x1:x_train})

        # 训练模型
        kneigh = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski')
        clf = kneigh.fit(train_out, y_train)
        y_test_pred = clf.predict(test_out)
        y_prob = clf.predict_proba(test_out)
        print("loop:%d, Pred label:%d  True label:%d, Prob: %f" % (loop, y_test_pred, y_test, y_prob[0][1]))
        label_pred_all.append(y_test_pred)
        label_true_all.extend(list(y_test))
        pred_prob.extend(y_prob)
        writer.close()

    return y_test, y_test_pred, y_prob, FCsZ_selected


if __name__ == "__main__":
    localtime = time.asctime(time.localtime(time.time()))
    print('Start Time: %s' % localtime)
    
    loo = LeaveOneOut()
    results = []
    pool = Pool(2)
    loop = 0
    for train_index, test_index in loo.split(data):
        loop = loop + 1
        result = pool.apply_async(mainLoop, args = [train_index, test_index, data, ylabel, loop])
        results.append(result)
    pool.close()
    pool.join()
        # result = mainLoop(train_index, test_index, data, ylabel, loop)
        # results.append(result)
        
    result_ytest_init = []
    result_ytest_pred = []
    prob_result = []
    FCsZ_select_concensus_loop = []

    for i in results:
        if i.ready():  # 进程函数是否已经启动了
            if i.successful():  # 进程函数是否执行成功
                a = i.get()
                result_ytest_init.append(a[0])
                result_ytest_pred.append(a[1])
                prob_result.append(list(a[2]))

    result_ytest_init = [i for item in result_ytest_init for i in item]
    result_ytest_pred = [i for item in result_ytest_pred for i in item]

    ## 评价
    print("********************************")
    accTest = accuracy_score(result_ytest_init, result_ytest_pred)
    print("Test accuracy: %f" %accTest)
    print("********************************")

    localtime = time.asctime(time.localtime(time.time()))
    print('End Time: %s' % localtime)

    # file_path = r'./output.xlsx'
    # writer = pd.ExcelWriter(file_path)
    # df = pd.DataFrame(result_ytest_init)
    # df.to_excel(writer, header=False, index=None, encoding='utf-8',sheet_name='Sheet1')

    # df = pd.DataFrame(result_ytest_pred)
    # df.to_excel(writer, header=False, index=None, encoding='utf-8',sheet_name='Sheet2')

    # df = pd.DataFrame(prob_result)
    # df.to_excel(writer, header=False, index=None, encoding='utf-8',sheet_name='Sheet3')

    # writer.save()
