import numpy as np

def trainModel(function_weights, alpha, imagesTraining, param): # zwraca wytrenowane wagi
    counter=0
    b=0
    while counter<=2:
        for i in range(len(imagesTraining)):
            realValue = param[i]

            y = np.dot(imagesTraining[i], function_weights)
            if y * realValue >= 1:  # kiedy dobrze sklasyfikowało
                # print('Well classified')
                result = [function_weights[j] - 2 * alpha * (1 / (100000)) * function_weights[j] for j in
                          range(len(function_weights))]
                function_weights = result

            else:
                # print('Bad classification')
                result = [function_weights[j] + alpha * (
                        realValue * (imagesTraining[i][j]) - 2 * (1 / (100000)) * function_weights[j]) for j
                          in range(len(imagesTraining[i]))]
                function_weights = result
                b+=1-y*realValue
        counter+=1
    bias=b/(counter*len(imagesTraining))
    return function_weights,bias
