import matplotlib.pyplot as img
import cv2 as cv
import random

def matrixImage(xImg, letter):
    data = img.imread(xImg)
    writeindataset(data, letter)
    return data

def writeindataset(matrice, letter):
    chemin = "dataImzml/train/"
    fichier = open(chemin + "dataset.txt", "a") 
    print(matrice.shape)
    prob224 = cv.resize(matrice, (224, 56))
    print(prob224.shape)
    # data resize
    prob224_reshape = prob224.reshape((1, 50176))
    fichier.write(letter)
    for val in prob224_reshape[0]:
        fichier.write("," + str(val))  
    fichier.write("\n")  
    fichier.close

matrixImage('dataImzml/train/img1.png', '1')


def shuffle() :
    #shuffle the lines of our file:
    lines = open(chemin+'dataset.txt').readlines() 
    random.shuffle(lines)
    open(chemin+'dataset.txt', 'w').writelines(lines) 
    print("The dataset is ready")
