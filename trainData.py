import proceessing as p
from os import listdir
from PIL import Image
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import cv2
import pickle

# crop images according size and sky
print("Cropping......................")
dirs = listdir("./datasets")
for dir in dirs:
    files = listdir("./datasets/"+str(dir))
    for file in files:
        p.resize_image(100, "./datasets/"+str(dir)+"/"+str(file), "./croppedImg/"+str(dir)+"/", str(file))
print("Cropped.......................")
train_data = []
train_label = []
# train data 
print("Training...........................")
croppeddirs = listdir("./croppedImg")
for dir in croppeddirs:
    files = listdir("./croppedImg/"+str(dir))
    for file in files:
        image = Image.open("./croppedImg/"+str(dir)+"/"+str(file))
        image = np.array(image)
        train_data.append(image)
        train_label.append(str(dir))
print("Save as data in npy...................")
train_data, train_label = p.shuffle(train_data, train_label)
np.save("train_data.npy",np.array(train_data))  # model root to save image models(image)
np.save("train_label.npy",np.array(train_label)) 
print("Saved npy files...................")

data = np.load("./train_data.npy")
data_label = np.load("train_label.npy")
data_label = np.reshape(data_label, (np.shape(data)[0],))
split_size = 10
kf = KFold(n_splits=split_size, shuffle=True)
kf.get_n_splits(data)

overallAccuracies = np.zeros(5)
generalOverallAccuracy = 0

for train_index, test_index in kf.split(data):
    # print("TRAIN:", train_index, "TEST:", test_index)
    train_data, test_data = data[train_index], data[test_index]
    train_label, test_label = data_label[train_index], data_label[test_index]

    # puts all features into a single array
    train_data_reshaped = train_data.reshape((len(train_data), -1))
    test_data_reshaped = test_data.reshape((len(test_data), -1))

    # Create a random forest Classifier. By convention, clf means 'Classifier'
    clf = RandomForestClassifier(bootstrap=False,
                                 max_leaf_nodes=None,
                                 n_estimators=12,  # The number of trees in the forest
                                 min_weight_fraction_leaf=0.0,
                                 )

    # Train the Classifier to take the training features and learn how they relate to the training(the species)
    clf.fit(train_data_reshaped, train_label)
    filename = 'finalized_model.pkl'
    pickle.dump(clf, open(filename, 'wb'))
print("Trained Data ........................................")
print("Saved Model(pkl)...........................................")
# inputImageName = "0.jpg"
# p.resize_image(100,"./test/"+str(inputImageName),"./test", inputImageName)
# input = cv2.imread("./test/"+str(inputImageName))
# input = input.reshape(1,-1)
# result = clf.predict(input)
# print(result)
    