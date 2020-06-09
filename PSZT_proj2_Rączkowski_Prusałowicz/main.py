import struct as st
from array import array
import nonlinear_model
import linear_model
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score

import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def readTrainingData(filePath): # wczytuje obrazki- zwraca tablice obrazków
    train_imagesfile = open(filePath, 'rb')
    train_imagesfile.seek(0)
    magic = st.unpack('>4B', train_imagesfile.read(4))  # .read(4) czytamy 4 bajty
    nImg = st.unpack('>I', train_imagesfile.read(4))[0]  # num of images
    nR = st.unpack('>I', train_imagesfile.read(4))[0]  # num of rows
    nC = st.unpack('>I', train_imagesfile.read(4))[0]  # num of column
    images_array = np.zeros((nImg, nR * nC))
    nBytesTotal = nImg * nR * nC * 1  # since each pixel data is 1 byte
    images_array = float(255) - np.asarray(
        st.unpack('>' + 'B' * nBytesTotal, train_imagesfile.read(nBytesTotal))).reshape(
        (nImg, nR * nC))
    return np.array(images_array), nR, nC


def readTrainingLabels(filePath): # wczytuje labele
    with open(filePath, 'rb') as file:
        magic, size = st.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
        labels = array("B", file.read())
        return labels


def convertLabels(labelsTraining, param): # konwertuje labele z {0,1,2,3...9} => {-1,1}
    convertedLabels = []
    for i in range(len(labelsTraining)):
        if labelsTraining[i] == param:
            convertedLabels.append(1)
        else:
            convertedLabels.append(-1)
    return convertedLabels


def divideIntoModels(labelsTraining):  # do taktyki one vs all => np jest 0 vs reszta to dla 0 jest label 1 a dla reszty 0
    models = []  # lista list gdzie element takiej listy to labele dla danej cyfry
    for i in range(10):
        models.append(convertLabels(labelsTraining, i))
    return models


def divideonevsonemodel(labelsTraining):  # do taktyki one vs one - tworzy listę list-> 1 lista przetrzymuje indeksy przykładów
    #które mają label 0
    groupedLabels = [[] for i in range(10)]
    for i in range(len(labelsTraining)):
        groupedLabels[labelsTraining[i]].append(i)
    return groupedLabels


def predict(feature_vector, weights,bias):
    y = np.dot(feature_vector, weights) +bias
    if y > 1:
        return 1
    else:
        return -1


def validate_model(feature, weights,bias): # walidacja - im punkt dalej od granicy to tym lepszy
    return (np.dot(feature, weights)) / (np.linalg.norm(weights))

def getImages(label1, label2, groupedLabels,imagesTraining):  # dzieli dane na obrazki z label1 i label2 i nadajemy odpowiednie labele do modelu
    images = []
    convertedLabels = []
    i = 0
    j = 0
    while i < len(groupedLabels[label1]) or j < len(groupedLabels[label2]):
        if i != len(groupedLabels[label1]):
            convertedLabels.append(1)
            images.append(imagesTraining[groupedLabels[label1][i]])
            i += 1
        if j != len(groupedLabels[label2]):
            convertedLabels.append(-1)
            images.append(imagesTraining[groupedLabels[label2][j]])
            j += 1
    return images, convertedLabels



def main():

    imagesTraining, rows, columns = readTrainingData('samples/train-images.idx3-ubyte')  # images[0] to wektor pikseli pierwszego obrazka
    labelsTraining = readTrainingLabels('samples/train-labels.idx1-ubyte')
    imagesPrediction, r, c = readTrainingData('samples/t10k-images.idx3-ubyte')
    labelsPrediction = readTrainingLabels('samples/t10k-labels.idx1-ubyte')
    probes = len(imagesPrediction)
    alpha=0.001
    for i in range(len(imagesTraining)):  # normalizacja DO WARTOŚCI (0,1)
        imagesTraining[i] = imagesTraining[i] / float(255)
    for i in range(len(imagesPrediction)):  # normalizacja
        imagesPrediction[i] = imagesPrediction[i] / float(255)
    weights_proper = np.zeros(rows * columns)  # inicjalizacja wag


    data=int(input('Choose options: \n 0. Quit program \n 1. Linear- one vs all \n 2. Linear- one vs one \n 3. Nonlinear (kernel:quadratic)- one vs all \n 4. Nonlinear (kernel:quadratic)- one vs one \n '
                   '5. Linear (Sklearn)- one vs all \n 6. Linear (Sklearn)- one vs one \n 7. Nonlinear (Sklearn, kernel:quadratic)- one vs all \n 8. Nonlinear (Sklearn, kernel:quadratic)- one vs one \n'))
    print("Ok")

    if data==0:
        exit(0)
    elif data == 1 or data == 3:

        nModels = divideIntoModels(labelsTraining)  # nmodels[0] to tam gdzie 0 są oznaczone jako 1 a reszta jako 0
        trained_weights = []
        trained_biases=[]
        for i in range(len(nModels)): # trenowanie modeli
             if data == 1:
                weights,bias=linear_model.trainModel(weights_proper, alpha, imagesTraining, nModels[i])
             elif data == 3:
                 weights, bias = nonlinear_model.fit(imagesTraining, nModels[i])

             trained_weights.append(weights)
             trained_biases.append(bias)
             print('Trained model nr: ', i + 1)

        correct = 0
        validations = []  # ocena prawdopodobieństwa słuszności każdego modelu
        for j in range(probes): # predykcja dla modelu one vs all
            for i in range(len(trained_weights)):
                predict(imagesPrediction[j], trained_weights[i],trained_biases[i])
                validations.append(validate_model(imagesPrediction[j], trained_weights[i],trained_biases[i]))

            index = validations.index(max(validations))
            print('Predicted by model: ', index)
            print('Real value: ', labelsPrediction[j])
            if labelsPrediction[j] == index:
                correct = correct + 1
            validations.clear()

        accuracy = correct / probes
        print('Accuracy: ', accuracy)

    elif data == 2 or data == 4:
        groupedLabels = divideonevsonemodel(labelsTraining)

        onevsoneweights = [[[] for j in range(10)] for i in range(10)]
        onevsonebiases = [[[] for j in range(10)] for i in range(10)]
        counter_models = 1
        alpha = 0.001
        for i in range(10):
            for j in range(i + 1, 10):
                dividedImages, convertedLabels = getImages(i, j, groupedLabels, imagesTraining)
                if data == 2:
                    trained_weight, trained_bias = linear_model.trainModel(weights_proper, alpha, dividedImages, convertedLabels)
                elif data == 4:
                    trained_weight, trained_bias = nonlinear_model.fit(dividedImages, convertedLabels)

                onevsoneweights[i][j].append(
                    trained_weight)  # tutaj przykład: i=0 j=1-> w [i][j] trzymamy wagi wytrenowane do modelu wyróżniającego 0 od 1
                onevsonebiases[i][j].append(trained_bias)
                onevsoneweights[j][i].append(trained_weight)
                onevsonebiases[j][i].append(trained_bias)
                print('Trained model nr ', counter_models)
                counter_models += 1

        new_validations = []
        for i in range(10):
            new_validations.append(0)
        new_counter = 0
        for k in range(probes):
            for i in range(10):
                for j in range(i + 1, 10):
                    prediction = predict(imagesPrediction[k], onevsoneweights[i][j][0], onevsonebiases[i][j][0])
                    change = sigmoid(
                        validate_model(imagesPrediction[k], onevsoneweights[i][j][0], onevsonebiases[i][j][0]))
                    if prediction == 1:
                        new_validations[i] += 10 + change

                    else:
                        new_validations[j] += 10 + change

            print('Validations: ', new_validations)
            index = new_validations.index(max(new_validations))
            print('Predicted by model: ', index)
            print('Real value: ', labelsPrediction[k])
            if index == labelsPrediction[k]:
                new_counter += 1
            new_validations = [0 for i in range(10)]

        print('Accuracy: ', (new_counter) / probes)

    elif data == 5:  # Linear (Sklearn)- one vs all

        clf = SVC(kernel='linear', C=0.1, gamma=0.01, decision_function_shape='ova')
        clf.fit(imagesTraining, labelsTraining)
        y_pred = clf.predict(imagesPrediction)
        accuracy = accuracy_score(y_true=list(labelsPrediction), y_pred=list(y_pred))
        print(clf, "Accuracy: {:.2f}%".format(accuracy * 100))


    elif data == 6:  # Linear (Sklearn)- one vs one
        clf = SVC(kernel='linear', C=0.1, gamma=0.01, decision_function_shape='ovo')
        clf.fit(imagesTraining, labelsTraining)
        y_pred = clf.predict(imagesPrediction)
        accuracy = accuracy_score(y_true=list(labelsPrediction), y_pred=list(y_pred))
        print(clf, "Accuracy: {:.2f}%".format(accuracy * 100))


    elif data == 7:  # Nonlinear (Sklearn, kernel:quadratic)- one vs all
        clf = SVC(kernel='poly', C=0.1, gamma=0.01, decision_function_shape='ova')
        clf.fit(imagesTraining, labelsTraining)
        y_pred = clf.predict(imagesPrediction)
        accuracy = accuracy_score(y_true=list(labelsPrediction), y_pred=list(y_pred))
        print(clf, "Accuracy: {:.2f}%".format(accuracy * 100))


    elif data == 8:  # Nonlinear (Sklearn, kernel:quadratic)- one vs one
        clf = SVC(kernel='poly', C=0.1, gamma=0.01, decision_function_shape='ovo')
        clf.fit(imagesTraining, labelsTraining)
        y_pred = clf.predict(imagesPrediction)
        accuracy = accuracy_score(y_true=list(labelsPrediction), y_pred=list(y_pred))
        print(clf, "Accuracy: {:.2f}%".format(accuracy * 100))

if __name__== "__main__":
    main()

