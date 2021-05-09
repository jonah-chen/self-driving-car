import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, BatchNormalization, Dropout, ReLU, Input, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.data import Dataset

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

from dirs import generate_train, generate_test

from time import time

def conv_block(x, kernels):
    y = Conv2D(kernels, (3, 3), padding="same")(x)
    y = BatchNormalization()(y)
    y = ReLU()(y)
    y = Conv2D(kernels, (3, 3), padding="same")(y)
    y = BatchNormalization()(y)
    y = ReLU()(y)
    return y

def build_model():
    inputs = Input(shape=(256,512,3,))
    c1 = conv_block(inputs, 64)
    t = MaxPooling2D((2,2))(c1)
    t = Dropout(0.2)(t)

    c2 = conv_block(t, 128)
    t = MaxPooling2D((2,2))(c2)
    t = Dropout(0.2)(t)

    c3 = conv_block(t, 256)
    t = MaxPooling2D((2,2))(c3)
    t = Dropout(0.2)(t)

    c4 = conv_block(t, 512)
    t = MaxPooling2D((2,2))(c4)
    t = Dropout(0.2)(t)

    c5 = conv_block(t, 1024)
    
    s = Conv2DTranspose(512, (2,2), strides=(2,2), padding="same")(c5)
    t = Add()([s, c4])
    t = Dropout(0.2)(t)
    t = conv_block(t, 512)

    s = Conv2DTranspose(256, (2,2), strides=(2,2), padding="same")(t)
    t = Add()([s, c3])
    t = Dropout(0.2)(t)
    t = conv_block(t, 256)

    s = Conv2DTranspose(128, (2,2), strides=(2,2), padding="same")(t)
    t = Add()([s, c2])
    t = Dropout(0.2)(t)
    t = conv_block(t, 128)

    s = Conv2DTranspose(64, (2,2), strides=(2,2), padding="same")(t)
    t = Add()([s, c1])
    t = Dropout(0.2)(t)
    t = conv_block(t, 64)
    
    t = Conv2D(30, (1, 1), activation="sigmoid")(t)

    return Model(inputs=[inputs], outputs=[t])

if __name__ == '__main__':
    model = tf.keras.models.load_model("model.h5")
    model.compile(optimizer=Adam(lr=1e-4), loss=categorical_crossentropy, metrics=['accuracy'])
    model.summary()

    xgen = Dataset.from_generator(generate_train, output_signature=(tf.TensorSpec(shape=(None,256,512,3),dtype=tf.float32), tf.TensorSpec(shape=(None,256,512,30), dtype=tf.float32)))

    ygen = Dataset.from_generator(generate_test, output_signature=(tf.TensorSpec(shape=(None,256,512,3),dtype=tf.float32), tf.TensorSpec(shape=(None,256,512,30), dtype=tf.float32)))

    tb = TensorBoard(log_dir="LOGS/1620358639.7773707", histogram_freq=1, profile_batch="500,520")
    mcpt = ModelCheckpoint("ckpt.h5", monitor="val_loss", save_best_only=True)

    model.fit(x=xgen, epochs=30, use_multiprocessing=True, workers=12, validation_data=ygen, callbacks=[mcpt, tb], initial_epoch=15)
    model.save("model.h5")

