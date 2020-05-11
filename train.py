import os, cv2, skimage
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from random import shuffle
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, Add, GlobalAveragePooling2D, DepthwiseConv2D, BatchNormalization, LeakyReLU
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau

train_len = 87000
batch_size = 64
imageSize = 64
target_dims = (imageSize, imageSize, 3)
num_classes = 29

train_dir = "asl_alphabet_train/"
classes = ['A','B','C','D','E','F',
           'G','H','I','J','K','L',
           'M','N','O','P','Q','R',
           'S','T','U','V','W','X',
           'Y','Z','delete','nothing','space']

def get_data(folder):
    X = np.empty((train_len, imageSize, imageSize, 3), dtype=np.float32)
    y = np.empty((train_len,), dtype=np.int)
    
    cnt = 0
    for folderName in os.listdir(folder):
        if not folderName.startswith('.'):
            label = 29
            try:
                label = classes.index(folderName)
            except ValueError:
                print('invalid folder')        
            for image_filename in os.listdir(folder + folderName):
                if cnt >= 87000:
                    break
                img_file = cv2.imread(folder + folderName + '/' + image_filename)
                if img_file is not None:
                    img_file = skimage.transform.resize(img_file, (imageSize, imageSize, 3))
                    img_arr = np.asarray(img_file).reshape((-1, imageSize, imageSize, 3))
                    X[cnt] = img_arr
                    y[cnt] = label
                    cnt += 1
    return X, y

X_train, y_train = get_data(train_dir)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1) 

y_trainHot = to_categorical(y_train, num_classes=num_classes)
y_testHot = to_categorical(y_test, num_classes=num_classes)

train_image_generator = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

val_image_generator = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
)

train_generator = train_image_generator.flow(x=X_train, y=y_trainHot, batch_size=batch_size, shuffle=True)
val_generator = val_image_generator.flow(x=X_test, y=y_testHot, batch_size=batch_size, shuffle=False)

inputs = Input(shape=target_dims)
net = Conv2D(32, kernel_size=3, strides=1, padding="same")(inputs)
net = LeakyReLU()(net)
net = Conv2D(32, kernel_size=3, strides=1, padding="same")(net)
net = LeakyReLU()(net)
net = Conv2D(32, kernel_size=3, strides=2, padding="same")(net)
net = LeakyReLU()(net)
net = Conv2D(32, kernel_size=3, strides=1, padding="same")(net)
net = LeakyReLU()(net)
net = Conv2D(32, kernel_size=3, strides=1, padding="same")(net)
net = LeakyReLU()(net)
net = Conv2D(32, kernel_size=3, strides=2, padding="same")(net)
net = LeakyReLU()(net)
shortcut = net
net = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal')(net)
net = BatchNormalization(axis=3)(net)
net = LeakyReLU()(net)
net = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal')(net)
net = BatchNormalization(axis=3)(net)
net = LeakyReLU()(net)
net = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal')(net)
net = BatchNormalization(axis=3)(net)
net = LeakyReLU()(net)
net = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal')(net)
net = BatchNormalization(axis=3)(net)
net = LeakyReLU()(net)
net = Add()([net, shortcut])
net = GlobalAveragePooling2D()(net)
net = Dropout(0.2)(net)

net = Dense(128, activation='relu')(net)
outputs = Dense(num_classes, activation='softmax')(net)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])

model.fit_generator(train_generator, epochs=10, validation_data=val_generator,
    steps_per_epoch=train_generator.__len__(),
    validation_steps=val_generator.__len__(),
    callbacks=[
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=5, verbose=1, mode='auto')
    ]
)

model.save("my_model.h5")
model_json = model.to_json()
with open("model2.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model2.h5")
print("Saved model to disk")