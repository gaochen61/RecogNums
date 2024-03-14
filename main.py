import numpy as np
import cv2 as cv
import gzip

def read_file(filename):
    f= gzip.open(filename=filename, mode="rb")
    content = f.read()
    buf=np.frombuffer(content,dtype=np.uint8)
    return buf

if __name__ == "__main__":
    # 读取文件中的图片，放到矩阵中
    x_train=read_file("./train-images-idx3-ubyte.gz")[16:].reshape((-1, 28, 28))
    y_train=read_file("./train-labels-idx1-ubyte.gz")[8:]
    x_test=read_file("./t10k-images-idx3-ubyte.gz")[16:].reshape((-1, 28, 28))
    y_test = read_file("./t10k-labels-idx1-ubyte.gz")[8:]

    #显示一张图片和这张图片的标签
    # cv.imshow("win",x_train[0])
    # print(y_train[0])
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    #计算每张图片的hog描述子
    hog = cv.HOGDescriptor((28,28), (14,14), (7,7), (7,7), 9, 1,-1, 0,0.2,True, 64, True)
    hog_descriptors=[]
    for img in x_train:
        hog_descriptors.append(hog.compute(img))
    hog_descriptors=np.squeeze(hog_descriptors)

    #获得svm模型
    model = cv.ml.SVM_create()
    model.setGamma(0.50625)
    model.setC(12.5)
    model.setKernel(cv.ml.SVM_RBF)
    model.setType(cv.ml.SVM_C_SVC)

    #进行训练
    train_labels = np.array(y_train, dtype=np.int32)
    print("begin training")
    model.trainAuto(hog_descriptors, cv.ml.ROW_SAMPLE, train_labels, 10)
    print("end training")
    model.save('svm.xml')
    hog.save('hog_descriptor.xml')

    #计算测试用图片的hog描述子
    hog_descriptors_test = []
    for img in x_test:
        hog_descriptors_test.append(hog.compute(img))
    hog_descriptors_test = np.squeeze(hog_descriptors_test)

    #判断训练成果
    test_labels = np.array(y_test, dtype=np.int32)
    predictions=model.predict(hog_descriptors_test)[1].ravel()
    accuracy=(test_labels==predictions).mean()
    print('Accuracy: %.2f %%' % (accuracy * 100))