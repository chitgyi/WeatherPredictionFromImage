import pickle
import cv2
import proceessing as p

with open('finalized_model.pkl', 'rb') as f:
    model = pickle.load(f)
    #input image name which exists in test folder
    inputImageName = "4.jpg"
    p.resize_image(100,"./test/"+str(inputImageName),"./test", inputImageName)
    input = cv2.imread("./test/"+str(inputImageName))
    input = input.reshape(1,-1)
    result = model.predict(input)
    print("Result --> ", result[0])