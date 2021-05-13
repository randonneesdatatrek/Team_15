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

    
def load_dataset(dataset_file_path):
    a = np.loadtxt(dataset_file_path, delimiter=',', converters={ 0 : lambda ch : ord(ch)-ord('a') }) #trier le fichier en separant sur chaque ligne chaque élément à l'aide de ','
    samples, letters = a[:,1:], a[:,0] #letters are the first element on each line, sample are the reste of the line
    return samples, letters


def loaded(chemin, dataset) :
    [samples, letters] = load_dataset(chemin+dataset)
    train_ratio = 0.6
    n_train_samples = int(len(samples) * train_ratio) 
    x_train, y_train = samples[:n_train_samples], letters[:n_train_samples]
    x_val, y_val = samples[n_train_samples:], letters[n_train_samples:]
    return train_ratio, n_train_samples, x_train, x_val, y_train, y_val
