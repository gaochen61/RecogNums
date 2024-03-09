import numpy as np
import requests
import gzip
import os
import hashlib
import cv2

# Load data
DATA_PATH = 'data/'


def LoadData(url):
    fp = os.path.join(DATA_PATH, hashlib.md5(url.encode('utf-8')).hexdigest())
    if os.path.isfile(fp):
        with open(fp, "rb") as f:
            data = f.read()
    else:
        with open(fp, "wb") as f:
            data = requests.get(url).content
            f.write(data)
    return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()


def CreateHOGDescriptor():
    winSize = (28, 28)
    blockSize = (14, 14)
    blockStride = (7, 7)
    cellSize = (7, 7)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradient = True

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture,
                            winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradient)
    return hog


def InitSVM(C=12.5, gamma=0.50625):
    model = cv2.ml.SVM_create()
    model.setGamma(gamma)
    model.setC(C)
    model.setKernel(cv2.ml.SVM_RBF)
    model.setType(cv2.ml.SVM_C_SVC)
    return model


def TrainSVM(model, samples, responses, kFold=10):
    model.trainAuto(samples, cv2.ml.ROW_SAMPLE, responses, kFold)
    return model


def SVMPredict(model, samples):
    return model.predict(samples)[1].ravel()


def EvaluateSVM(model, samples, labels):
    predictions = SVMPredict(model, samples)
    accuracy = (labels == predictions).mean()
    print('Accuracy: %.2f %%' % (accuracy*100))

    confusion = np.zeros((10, 10), np.int32)
    for i, j in zip(labels, predictions):
        confusion[int(i), int(j)] += 1
    print('confusion matrix:')
    print(confusion)


if __name__ == "__main__":
    # 加载数据集
    X_train = LoadData(
        "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
    Y_train = LoadData(
        "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
    X_test = LoadData(
        "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
    Y_test = LoadData(
        "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]

    print("The data is loaded.")

    # 提取训练集的HOG特征
    hog = CreateHOGDescriptor()
    hog_descriptors = []
    for img in X_train:
        hog_descriptors.append(hog.compute(img))

    hog_descriptors = np.squeeze(hog_descriptors)

    print('Traning svm classifier...')
    # 加载SVM分类器模型进行训练
    svm_model = InitSVM()
    train_labels = np.array(Y_train, dtype=np.int32)
    TrainSVM(svm_model, hog_descriptors, train_labels)

    # 训练完后保存模型
    svm_model.save('svm.xml')
    hog.save('hog_descriptor.xml')
    print('The model are saved to xml file.')

    # 提取测试集的HOG特征
    hog_descriptors_test = []
    for img in X_test:
        hog_descriptors_test.append(hog.compute(img))
    hog_descriptors_test = np.squeeze(hog_descriptors_test)

    print('Evaluating model...')
    # 在测试集上测试模型的准确率并打印混淆矩阵
    test_labels = np.array(Y_test, dtype=np.int32)
    EvaluateSVM(svm_model, hog_descriptors_test, test_labels)
    print('Done.')
