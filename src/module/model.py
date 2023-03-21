import tensorflow as tf
from keras.models import Model
from keras.applications import EfficientNetB0, ResNet50, VGG16, Xception, MobileNet, DenseNet121, NASNetLarge, InceptionV3


def cnnModel(target_size):
    def branch(shape, branch_num):
        input_shape = tf.keras.layers.Input(shape=(shape[0], shape[1], shape[2]))
        x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')(input_shape)
        x = tf.keras.layers.MaxPool2D(2, 2)(x)
        x =  tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
        x = tf.keras.layers.MaxPool2D(2, 2)(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = tf.keras.layers.MaxPool2D(2, 2)(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = tf.keras.layers.MaxPool2D(2, 2)(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = tf.keras.layers.MaxPool2D(2, 2)(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        # for layer in x.layers:
        #     layer._name = layer._name + str(branch_num)
        model = Model(inputs = input_shape, outputs = x)
        return model
    
    b1 = branch(target_size, '_1')
    b2 = branch(target_size, '_2')
    b3 = branch(target_size, '_3')
    concat_model = tf.keras.layers.concatenate([b1.output, b2.output, b3.output])
    concat_model = tf.keras.layers.Dense(256, activation = "relu")(concat_model)
    concat_model = tf.keras.layers.BatchNormalization()(concat_model)
    concat_model = tf.keras.layers.Dropout(0.5)(concat_model)
    concat_model = tf.keras.layers.Dense(128, activation = "relu")(concat_model)
    concat_model = tf.keras.layers.BatchNormalization()(concat_model)
    concat_model = tf.keras.layers.Dropout(0.5)(concat_model)
    concat_output = tf.keras.layers.Dense(1, activation='sigmoid')(concat_model)

    total_model = Model(inputs = [b1.input, b2.input, b3.input], outputs = [concat_output])
    total_model._name = 'CNN'
    return total_model

def vgg16Model(target_size):
    def branch(shape, branch_num):
        input_shape = tf.keras.layers.Input(shape=(shape[0], shape[1], shape[2]))
        training_model = VGG16(weights = None, include_top = False, input_tensor = input_shape)
        training_model.trainable = False
        x = training_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        for layer in training_model.layers:
            layer._name = layer._name + str(branch_num)
        model = Model(inputs = training_model.input, outputs = x)
        return model
    b1 = branch(target_size, '_1')
    b2 = branch(target_size, '_2')
    b3 = branch(target_size, '_3')
    concat_model = tf.keras.layers.concatenate([b1.output, b2.output, b3.output])
    concat_model = tf.keras.layers.Dense(256, activation = "relu")(concat_model)
    concat_model = tf.keras.layers.BatchNormalization()(concat_model)
    concat_model = tf.keras.layers.Dropout(0.5)(concat_model)
    concat_model = tf.keras.layers.Dense(128, activation = "relu")(concat_model)
    concat_model = tf.keras.layers.BatchNormalization()(concat_model)
    concat_model = tf.keras.layers.Dropout(0.5)(concat_model)
    concat_output = tf.keras.layers.Dense(1, activation='sigmoid')(concat_model)

    total_model = Model(inputs = [b1.input, b2.input, b3.input], outputs = [concat_output])
    total_model._name = 'VGG16'
    return total_model

def resnet50Model(target_size):
    def branch(shape, branch_num):
        input_shape = tf.keras.layers.Input(shape=(shape[0], shape[1], shape[2]))
        training_model = ResNet50(weights = None, include_top = False, input_tensor = input_shape)
        training_model.trainable = False
        x = training_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        for layer in training_model.layers:
            layer._name = layer._name + str(branch_num)
        model = Model(inputs = training_model.input, outputs = x)
        return model
    b1 = branch(target_size, '_1')
    b2 = branch(target_size, '_2')
    b3 = branch(target_size, '_3')
    concat_model = tf.keras.layers.concatenate([b1.output, b2.output, b3.output])
    concat_model = tf.keras.layers.Dense(256, activation = "relu")(concat_model)
    concat_model = tf.keras.layers.BatchNormalization()(concat_model)
    concat_model = tf.keras.layers.Dropout(0.5)(concat_model)
    concat_model = tf.keras.layers.Dense(128, activation = "relu")(concat_model)
    concat_model = tf.keras.layers.BatchNormalization()(concat_model)
    concat_model = tf.keras.layers.Dropout(0.5)(concat_model)
    concat_output = tf.keras.layers.Dense(1, activation='sigmoid')(concat_model)

    total_model = Model(inputs = [b1.input, b2.input, b3.input], outputs = [concat_output])
    total_model._name = 'ResNet50'
    return total_model
 

def xceptionModel(target_size):
    def branch(shape, branch_num):
        input_shape = tf.keras.layers.Input(shape=(shape[0], shape[1], shape[2]))
        training_model = Xception(weights = None, include_top = False, input_tensor = input_shape)
        training_model.trainable = False
        x = training_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        for layer in training_model.layers:
            layer._name = layer._name + str(branch_num)
        model = Model(inputs = training_model.input, outputs = x)
        return model
    b1 = branch(target_size, '_1')
    b2 = branch(target_size, '_2')
    b3 = branch(target_size, '_3')
    concat_model = tf.keras.layers.concatenate([b1.output, b2.output, b3.output])
    concat_model = tf.keras.layers.Dense(256, activation = "relu")(concat_model)
    concat_model = tf.keras.layers.BatchNormalization()(concat_model)
    concat_model = tf.keras.layers.Dropout(0.5)(concat_model)
    concat_model = tf.keras.layers.Dense(128, activation = "relu")(concat_model)
    concat_model = tf.keras.layers.BatchNormalization()(concat_model)
    concat_model = tf.keras.layers.Dropout(0.5)(concat_model)
    concat_output = tf.keras.layers.Dense(1, activation='sigmoid')(concat_model)

    total_model = Model(inputs = [b1.input, b2.input, b3.input], outputs = [concat_output])
    total_model._name = 'xception'
    return total_model

def nesnatlargeModel(target_size):
    def branch(shape, branch_num):
        input_shape = tf.keras.layers.Input(shape=(shape[0], shape[1], shape[2]))
        training_model = NASNetLarge(weights = None, include_top = False, input_tensor = input_shape)
        training_model.trainable = False
        x = training_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        for layer in training_model.layers:
            layer._name = layer._name + str(branch_num)
        model = Model(inputs = training_model.input, outputs = x)
        return model
    b1 = branch(target_size, '_1')
    b2 = branch(target_size, '_2')
    b3 = branch(target_size, '_3')
    concat_model = tf.keras.layers.concatenate([b1.output, b2.output, b3.output])
    concat_model = tf.keras.layers.Dense(256, activation = "relu")(concat_model)
    concat_model = tf.keras.layers.BatchNormalization()(concat_model)
    concat_model = tf.keras.layers.Dropout(0.5)(concat_model)
    concat_model = tf.keras.layers.Dense(128, activation = "relu")(concat_model)
    concat_model = tf.keras.layers.BatchNormalization()(concat_model)
    concat_model = tf.keras.layers.Dropout(0.5)(concat_model)
    concat_output = tf.keras.layers.Dense(1, activation='sigmoid')(concat_model)

    total_model = Model(inputs = [b1.input, b2.input, b3.input], outputs = [concat_output])
    total_model._name = 'inceptionV3'
    return total_model

def inceptionV3Model(target_size):
    def branch(shape, branch_num):
        input_shape = tf.keras.layers.Input(shape=(shape[0], shape[1], shape[2]))
        training_model = InceptionV3(weights = None, include_top = False, input_tensor = input_shape)
        training_model.trainable = False
        layer_dict = dict([(layer.name, layer) for layer in training_model.layers])
        x = training_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        for layer in training_model.layers:
            layer._name = layer._name + str(branch_num)
        model = Model(inputs = training_model.input, outputs = x)
        return model
    b1 = branch(target_size, '_1')
    b2 = branch(target_size, '_2')
    b3 = branch(target_size, '_3')
    concat_model = tf.keras.layers.concatenate([b1.output, b2.output, b3.output])
    concat_model = tf.keras.layers.Dense(256, activation = "relu")(concat_model)
    concat_model = tf.keras.layers.BatchNormalization()(concat_model)
    concat_model = tf.keras.layers.Dropout(0.5)(concat_model)
    concat_model = tf.keras.layers.Dense(128, activation = "relu")(concat_model)
    concat_model = tf.keras.layers.BatchNormalization()(concat_model)
    concat_model = tf.keras.layers.Dropout(0.5)(concat_model)
    concat_output = tf.keras.layers.Dense(1, activation='sigmoid')(concat_model)

    total_model = Model(inputs = [b1.input, b2.input, b3.input], outputs = [concat_output])
    total_model._name = 'inceptionV3'
    return total_model
 
def mobileNetModel(target_size):
    def branch(shape, branch_num):
        input_shape = tf.keras.layers.Input(shape=(shape[0], shape[1], shape[2]))
        training_model = MobileNet(weights = None, include_top = False, input_tensor = input_shape)
        training_model.trainable = False
        x = training_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        for layer in training_model.layers:
            layer._name = layer._name + str(branch_num)
        model = Model(inputs = training_model.input, outputs = x)
        return model
    b1 = branch(target_size, '_1')
    b2 = branch(target_size, '_2')
    b3 = branch(target_size, '_3')
    concat_model = tf.keras.layers.concatenate([b1.output, b2.output, b3.output])
    concat_model = tf.keras.layers.Dense(256, activation = "relu")(concat_model)
    concat_model = tf.keras.layers.BatchNormalization()(concat_model)
    concat_model = tf.keras.layers.Dropout(0.5)(concat_model)
    concat_model = tf.keras.layers.Dense(128, activation = "relu")(concat_model)
    concat_model = tf.keras.layers.BatchNormalization()(concat_model)
    concat_model = tf.keras.layers.Dropout(0.5)(concat_model)
    concat_output = tf.keras.layers.Dense(1, activation='sigmoid')(concat_model)

    total_model = Model(inputs = [b1.input, b2.input, b3.input], outputs = [concat_output])
    total_model._name = 'mobileNet'
    return total_model
 
