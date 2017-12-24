# -*- coding:utf8 -*-
from time import time

import nltk
import os
from sql import addData
from sql import creatTable

from sql import selectData
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist
import numpy as np
from sklearn.cluster import KMeans

np.set_printoptions(threshold='nan')
wordEngStop = nltk.corpus.stopwords.words('english')
english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '!', '|', '@', '#', '&', '\\', '%', '$', '*', '=',
                        '\\n', '\n', 'abstract=', '{', '}']


def spiltWord(rootdir):
    word_number = 0
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    for i in range(filerange[0], filerange[1]):
        path = os.path.join(rootdir, list[i])
        sec_list = os.listdir(path)  # 列出文件夹下所有的目录与文件
        for j in range(0, len(sec_list)):
            sec_path = os.path.join(path, sec_list[j])
            print sec_path
            third_list = os.listdir(sec_path)  # 列出文件夹下所有的目录与文件
            for k in range(0, len(third_list)):
                third_path = os.path.join(sec_path, third_list[k])
                if os.path.isfile(third_path):
                    with open(third_path, 'r') as fin:
                        for eachLine in fin:
                            eachLine = eachLine.lower().decode('utf-8', 'ignore')  # 小写
                            tokens = nltk.word_tokenize(eachLine)  # 分词（与标点分开）
                            for word in tokens:
                                if not word in english_punctuations:  # 去标点
                                    if not '\\' in word:
                                        if not '~' in word:
                                            if not '|' in word:
                                                if not word in wordEngStop:  # 去停用词
                                                    if addData(table_name, word):
                                                        word_number += 1
    return word_number

def setFile(rootdir, word_total):
    filenparr = np.zeros((file_total, word_total))  # 29500
    row = 0
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    for i in range(filerange[0], filerange[1]):
        path = os.path.join(rootdir, list[i])
        sec_list = os.listdir(path)  # 列出文件夹下所有的目录与文件
        for j in range(0, len(sec_list)):
            sec_path = os.path.join(path, sec_list[j])
            third_list = os.listdir(sec_path)  # 列出文件夹下所有的目录与文件
            for k in range(0, len(third_list)):
                third_path = os.path.join(sec_path, third_list[k])
                if os.path.isfile(third_path):
                    row += 1
                    print row
                    with open(third_path, 'r') as fin:
                        for eachLine in fin:
                            eachLine = eachLine.lower().decode('utf-8', 'ignore')  # 小写
                            tokens = nltk.word_tokenize(eachLine)  # 分词（与标点分开）
                            for word in tokens:
                                if not word in wordEngStop:  # 去停用词
                                    col = selectData('data', word)
                                    if col:
                                        filenparr[row][col] = 1

    np.save(filevetctor_npy, filenparr)
    return filenparr


def getRelatmatrix(nparr):
    relation_matrix = dist.squareform(dist.pdist(nparr, 'jaccard'))
    np.save(relation_npy, relation_matrix)


def getRBF(nparr, o):
    print "o:"
    print o
    N = np.exp(-((np.square(nparr)) / (2 * o * o)))
    return N


def getWbyKNN(dis_matrix, k):
    W = np.zeros((file_total, file_total))
    for index, each in enumerate(dis_matrix):
        index_array = np.argsort(each)
        W[index][index_array[1:k + 1]] = 1
    tmp_W = np.transpose(W)
    W = (tmp_W + W) / 2.0
    return W


def getLaplace(W):
    D = np.diag(np.zeros(file_total))
    for i in range(file_total):
        D[i][i] = sum(W[i])

    # 规范的拉普拉斯
    L = (D - W)

    # L = np.linalg.inv(np.power(D,0.5)) * L * np.linalg.inv(np.power(D,0.5))
    return L


# 特征值和对应特征向量
def getEigen(Laplace, cluster_num):
    eigvalue, eigvector = np.linalg.eig(Laplace)
    dim = len(eigvalue)
    # 得到3400*6个具有特征的小矩阵
    dictEigval = dict(zip(eigvalue, range(0, dim)))
    # 从小到大排序
    kEig = np.sort(eigvalue)[0:cluster_num]
    # 6个特征矩阵所在大矩阵的位置
    ix = [dictEigval[k] for k in kEig]
    return eigvector[:, ix]


def plot(narray):
    np.set_printoptions(threshold='nan')
    plt.title('file'+str(file_number)+'way'+str(way))

    plt.xlabel('k')
    plt.ylabel('o')
    plt.imshow(narray)
    plt.show()


def getOrigin_nparr():
    origin_nparr = []
    for key, value in enumerate(origin_labels[:-1]):
        temp = np.ones((value), dtype=np.int) * key
        if (origin_nparr == []):
            origin_nparr = temp
        else:
            origin_nparr = np.r_[origin_nparr, temp]
    temp = np.ones((origin_labels[-1]), dtype=np.int) * 0
    origin_nparr = np.r_[origin_nparr, temp]
    return origin_nparr


def mykMeans(nparr, cluster_num):
    #循环5次取最大值，较少随机性
    # label_list=[]
    # intertia_list=[]
    # for i in range(5):
    clf = KMeans(n_clusters=cluster_num,max_iter=1000,tol=1e-12)  # 设定k调用KMeans算法
    s = clf.fit(nparr)  # 加载数据集合
    labels = clf.labels_
    # label_list.append(labels)
    # 显示聚类效果
    # intertia_list.append(clf.inertia_)
    # print "中心点"
    # print centroids
    # print type(centroids)  # 显示中心点
    # print "效果"
    # print clf.inertia_  # 显示聚类效果
    # max_index=np.where(intertia_list==np.max(intertia_list))
    # return  label_list[3]
    return labels


def getDiffmatrix(origin_arr, predict_arr):
    '''
    :param origin_arr:
    :param predict_arr:
    :return a matrix of origin * predict  relation :
    such as
    3->0 ,5->1
    '''
    realtion_matrix = np.zeros((cluster_num, cluster_num))

    # 得到标签对应和矩阵
    diff_matrix = np.zeros((cluster_num, cluster_num))
    for origin, predict in zip(origin_arr, predict_arr):
        diff_matrix[origin][predict] += 1

    # 使用数独取最大值算法，找到初始标签和预测标签的关系
    temp_matrix = diff_matrix
    row_arr = []
    col_arr = []
    for i in range(cluster_num):
        raw_col = np.where(temp_matrix == np.max(temp_matrix))
        # 获取在原矩阵中最大值的位置
        row = np.where(diff_matrix == np.max(temp_matrix))[0]
        col = np.where(diff_matrix == np.max(temp_matrix))[1]
        # 避免获取重复的最大值的位置
        r = 0
        c = 0
        for r in row:
            if r not in row_arr:
                row_arr.append(r)
                break
        for c in col:
            if c not in col_arr:
                col_arr.append(c)
                break
        # 给矩阵赋值得到原始标签向量与预测标签向量的关系矩阵6*6
        '''
        [[0,1,0,0,0,0]
        [0,0,0,1,0,0]]
        表示原始标签1对应预测标签2，原始标签2对应预测标签3
        '''
        realtion_matrix[r][c] = 1
        # 去除最大值所在的行列
        temp_matrix = np.delete(temp_matrix, raw_col[0][0], axis=0)  # 删除B的第m行
        temp_matrix = np.delete(temp_matrix, raw_col[1][0], axis=1)  # 删除B的第n列


    return realtion_matrix


def getAccuracyrate(predict_arr, relation_matrix):
    '''
    :param origin_arr: 1*3400 [0,0,0,0,0....1,1,1,1...2,2,2]
    :param predict_arr:1*3400 [0,1,2,0,3....]
    :param relation_matrix: 6*6 [[0,1,0,0,0]] means the origin_label 0 match the predict_label 1
    if origin_arr[1]=0 match predict_arr[1]=1  '0 match 1'
    n+=1
    :return:  an int of rate  : n/N n is a number of correct match, N is a total number
    '''
    n = 0.0
    N = file_total
    # 关系矩阵中ori_label0,1,2,3,4,5分别对应的predict 4,5,2,0,3,1
    row_cols = np.where(relation_matrix == 1)
    cols = row_cols[1]
    # 遍历predict_arr与origin_labels对比，发现到比对成功的n+1
    # origin_label多了一个剩余文件数3
    start = 0
    for key, value in enumerate(origin_labels[:-1]):
        for j in range(start, start + value):
            if predict_arr[j] == cols[key]:
                n += 1
        start += value

    print "correct number: " + str(n)
    print "accuracy: " + str(n / N)
    return n / N


def plot3D(x, y, z):

    plt.grid(True)  # 是否网格化
    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    ax.set_title("accuracy")

    # 将数据点分成三部分画，在颜色上有区分度
    try:
        ax.scatter(x, y, z, c='r')  # 绘制数据点
        ax.set_zlabel('accuracy')  # 坐标轴
        ax.set_ylabel('O')
        ax.set_xlabel('K')
        plt.show()
    except:
        return


def saveResultmatrix(x, y, z):
    np.set_printoptions(threshold='nan')
    plt.title('file' + str(file_number) + 'way' + str(way))
    if (way==1):
        plt.xlabel('k')
        plt.ylabel('accuracy')
        plt.scatter(x,z)
        plt.show()
    else:
        listlen=len(z)
        temp = np.zeros((listlen, listlen))
        for i, v in enumerate(temp):
            temp[i] = z[i]
        plot(temp)
    # ox, oy, z = x, y, z
    # resultmatrix = z
    # lenth = len(ox)
    # x = []
    # y = []
    # for i in range(lenth):
    #     for j in range(lenth):
    #         x.append(ox[i])
    # for i in range(lenth):
    #     for j in range(lenth):
    #         y.append(oy[j])
    # resultmatrix = np.row_stack((resultmatrix, x))
    # resultmatrix = np.row_stack((resultmatrix, y))
    # np.save(result_npy, resultmatrix)
    #
    # plot3D(x, y, z)



if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    # 正则化规范化拉普拉斯
    origin1_labels = [481, 593, 585, 598, 594, 546,3]
    origin2_labels = [584, 591, 590, 578, 593, 4]

    file1_total = 3400
    file2_total = 2940

    cluster1_num = 6
    cluster2_num = 5

    start = time()
    # 文件夹
    file_number = 1
    # 1=直接使用jarcard系数,2使用o系数计算RBF
    way = 2
    # 系数长度
    listlen = 1
    # 数据库
    print ('A simple execute demo . when finished ,it will show a plot.')
    file_number = input('Select the data folder( 1 or 2 ):')
    way = input('The calculation method of weight selection(1 is to use Jaccard coefficient , 2 is to use RBF of Jaccard distance): ')
    listlen = input('The number of results that need to be calculated (1 (minimum)): ')
    database = input('if to regenerate the dictionary library(0 no(recommended) , 1 yes)): ')
    nltk.download('stopwords')
    print 'Task initiation'
    if file_number == 1:
        file_total = file1_total
        filerange = [0, 1]
        origin_labels = origin1_labels
        cluster_num = cluster1_num
        table_name = 'data1'
    else:
        file_total = file2_total
        filerange = [1, 2]
        cluster_num = cluster2_num
        origin_labels = origin2_labels
        table_name = 'data2'
    try:
        creatTable(table_name)
    except:
        print 'database existed'
    klist = np.linspace(50, file_total - 50, listlen, dtype=np.int)  # min 200 max3000
    olist = np.linspace(1e-2, 1e-6, listlen)  # min -7 max -4

    task_finish = 0
    ratelist = []

    result_npy = 'file' + str(file_number) + 'way' + str(way) + 'result.npy'
    relation_npy = 'file' + str(file_number) + 'way' + str(way) + 'relationmatrix.npy'
    N_npy = 'file' + str(file_number) + 'way' + str(way) + 'N.npy'
    rate_npy = 'file' + str(file_number) + 'way' + str(way) + 'rate.npy'
    filevetctor_npy = 'file' + str(file_number) + 'way' + str(way) + 'filevetctor.npy'
    rootdir = 'data'

    if (database == 1):
        word_total = spiltWord(rootdir)
        filevetcor = setFile(rootdir, word_total + 1)
        getRelatmatrix(filevetcor)
    if way == 1:
        task_total = len(klist)
        N = 1 - np.load(relation_npy)
        # templist = olist
        # for key,i in enumerate(templist):
        #     olist[key]=i*1e6
        for k in klist:
            print "k:"
            print k
            W = getWbyKNN(N, k)
            L = getLaplace(W)
            E = getEigen(L, cluster_num)
            predict_lable = mykMeans(E, cluster_num)
            relation_matrix = getDiffmatrix(getOrigin_nparr(), predict_lable)
            rate = getAccuracyrate(predict_lable, relation_matrix)
            ratelist.append(rate)
            # np.save(rate_npy, ratelist)
            task_finish += 1
            print('progress status:' + str(task_finish) + "/" + str(task_total))
    else:
        task_total = len(klist) * len(olist)
        for o in olist:
            N = getRBF(np.load(relation_npy), o)
            for k in klist:
                print "k:"
                print k
                W = getWbyKNN(N, k)
                L = getLaplace(W)
                E = getEigen(L, cluster_num)
                predict_lable = mykMeans(E, cluster_num)
                relation_matrix = getDiffmatrix(getOrigin_nparr(), predict_lable)
                rate = getAccuracyrate(predict_lable, relation_matrix)
                ratelist.append(rate)
                # np.save(rate_npy, ratelist)
                task_finish += 1
                print('progress status:' + str(task_finish) + "/" + str(task_total))
    saveResultmatrix(klist, olist, ratelist)
    stop = time()
    print('time-consuming:' + str(stop - start) + "second")
