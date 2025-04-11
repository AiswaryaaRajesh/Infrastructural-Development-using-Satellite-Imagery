import cv2                 
import numpy as np         
import os                  
from random import shuffle 
from tqdm import tqdm

TRAIN_DIR = 'training'
TEST_DIR = 'testing'

IMG_SIZE = 50
LR = 1e-3
MODEL_NAME = 'infra-{}-{}.model'.format(LR, '2conv-basic')


def label_img(img):
    word_label = img[0]
    print(word_label)
  
    if word_label == 'l':
        print('Low Infrastructure Development')
        return [1,0,0]
    
    elif word_label == 'h':
        print('High Infrastructure Development')
        return [0,1,0]
    
    elif word_label == 'm':
        print('Medium Infrastructure Development')
        return [0,0,1]
    
"""def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        print('##############')
        print(label)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data
"""

def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        print('##############')
        print(label)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: Unable to read {path}")
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        # Ensure label is a NumPy array
        label = np.array(label, dtype=np.float32)
        training_data.append([img, label])
    shuffle(training_data)
    # Convert to numpy array with dtype=object to avoid shape mismatch
    np.save('./train_data.npy', np.array(training_data, dtype=object))
    return training_data

def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        kp =sift.detect(img, None)
        img=cv2.drawKepoints(img, kp , None)
        testing_data.append([np.array(img), img_num])
        
    shuffle(testing_data)
    np.save('./test_data.npy', testing_data)
    return testing_data

train_data = create_train_data()



#Creating CNN
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
tf.get_logger().setLevel('ERROR')



convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

convnet = conv_2d(convnet, 32, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 64, 3, activation='relu')   # This is where it’s different
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 128, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 32, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)
convnet = fully_connected(convnet, 3, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')


#Load pre=trained model
if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('Model Loaded!')

train = train_data[:-25]
test = train_data[-25:]

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
Y = [i[1] for i in train]
print(X.shape)
test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
test_y = [i[1] for i in test]
print(test_x.shape)


#Trains for 100 epochs; Uses Adam optimizer with categorical cross-entropy loss.
#model will go through the entire training dataset 100 times.
model.fit({'input': X}, {'targets': Y},n_epoch=100, validation_set=({'input': test_x}, {'targets': test_y}),snapshot_step=20, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)
