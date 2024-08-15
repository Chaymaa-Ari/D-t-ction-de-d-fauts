# activier la fonction print de python 3
from __future__ import print_function 

import keras # bibliothéque pour entrainner les réseaux de neuronnes
import cv2
import numpy as np
from keras.layers import Input, Dense, Dropout, Activation, Concatenate, BatchNormalization, Flatten
from keras.models import Model
from keras.layers import Conv2D, GlobalAveragePooling2D, AveragePooling2D, ZeroPadding2D, MaxPooling2D
from keras.regularizers import l2
from PIL import Image, ImageOps
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
import math
##################################################################################################################################################################



# Fonction pour ajuster le taux d'apprentissage en chaque itération pendant l'entrainement
# A chaque étape on réduit cet valeur pour améliorer la convergence du modéle
def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.1
    epochs_drop = 7.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate


##################################################################################################################################################################



# On définit une architecture de réseau de neurones convolutifs qui relie chaque couche à toutes les autres couches dans un modèle dense

def DenseNet(input_shape=None, dense_blocks=3, dense_layers=-1, growth_rate=12, nb_classes=None, dropout_rate=None,
             bottleneck=False, compression=1.0, weight_decay=1e-4, depth=40):
    
    # vérification du nombre de classes 
    if nb_classes==None:
        raise Exception('Please define number of classes (e.g. num_classes=10). This is required for final softmax.')
    
    # vérification de la valeur compression .la plage valide entre 0 et 1. 
    # La compression est utilisée pour réduire le nombre de canaux entre les blocs denses.
    if compression <=0.0 or compression > 1.0:
        raise Exception('Compression have to be a value between 0.0 and 1.0. If you set compression to 1.0 it will be turn off.')
    """ 
    La compression dans le contexte de l'architecture DenseNet fait référence à la réduction du nombre de canaux de sortie entre les blocs denses. 
    Elle est utilisée pour contrôler la complexité du modèle en réduisant le nombre de paramètres et de calculs requis, tout en maintenant les
    performances du réseau.
    """
    
    
    # vérification du nombre de couches dans chaque bloc dense
    if type(dense_layers) is list:
        if len(dense_layers) != dense_blocks:
            raise AssertionError('Number of dense blocks have to be same length to specified layers')
    elif dense_layers == -1:
        if bottleneck:
            dense_layers = (depth - (dense_blocks + 1))/dense_blocks // 2
        else:
            dense_layers = (depth - (dense_blocks + 1))//dense_blocks
        dense_layers = [int(dense_layers) for _ in range(dense_blocks)]
    else:
        dense_layers = [int(dense_layers) for _ in range(dense_blocks)]
      
      
    # Définition de la couche d'entrée  
    img_input = Input(shape=input_shape)
    # Initialisation des canaux
    nb_channels = growth_rate * 2
    
    
    # Couches de convolutions initiales
    #ZeroPadding2D et Conv2D: Ajoute une couche de convolutions initiale avec un padding de 3 pixels, suivie par une couche de convolution avec un noyau de taille 7x7 et des filtres nb_channels.
    
    x = ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)  # Ajoute des zéros autour de l'image en entrée pour conserver les dimensions après la convolution
    x = Conv2D(nb_channels, (7,7),strides=2 , use_bias=False, kernel_regularizer=l2(weight_decay))(x)  #Applique une opération de convolution 2D à l'image. pour réduire la dimension
    
    # Ajoute une normalisation par lots suivie d'une activation ReLU.
    x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)#
    x = Activation('relu')(x)
    
    # Ajoute un padding de 1 pixel suivi d'une couche de mise en commun avec un pool de taille 3x3 et un stride de 2.
    x = ZeroPadding2D(padding=((1,1), (1, 1)))(x)
    x = MaxPooling2D(pool_size = (3, 3), strides = 2)(x) #
    
    
    
    # Construction des blocs denses
    for block in range(dense_blocks):
        
        # Add dense block
        x, nb_channels = dense_block(x, dense_layers[block], nb_channels, growth_rate, dropout_rate, bottleneck, weight_decay)
        
        if block < dense_blocks - 1:  # if it's not the last dense block
            # Add transition_block
            x = transition_layer(x, nb_channels, dropout_rate, compression, weight_decay)
            nb_channels = int(nb_channels * compression)
    x = AveragePooling2D(pool_size = 7)(x) 
    x = Flatten(data_format = 'channels_last')(x)
    x = Dense(nb_classes, activation='softmax', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(x)
    
    model_name = None
    if growth_rate >= 36:
        model_name = 'widedense'
    else:
        model_name = 'dense'
        
    if bottleneck:
        model_name = model_name + 'b'
        
    if compression < 1.0:
        model_name = model_name + 'c'
        
    return Model(img_input, x, name=model_name), model_name  # on retourne le modéle et son nom



##################################################################################################################################################################


def dense_block(x, nb_layers, nb_channels, growth_rate, dropout_rate=None, bottleneck=False, weight_decay=1e-4):

    x_list = [x]
    for i in range(nb_layers):
        cb = convolution_block(x, growth_rate, dropout_rate, bottleneck, weight_decay)
        x_list.append(cb)
        x = Concatenate(axis=-1)(x_list)
        nb_channels += growth_rate
    return x, nb_channels

""" 
cette fonction implémente la logique de construction d'un bloc dense dans DenseNet, où chaque couche génère des caractéristiques qui sont concaténées 
avec les caractéristiques générées par toutes les couches précédentes, et le nombre de canaux est mis à jour en conséquence.cette fonction implémente la 
logique de construction d'un bloc dense dans DenseNet, où chaque couche génère des caractéristiques qui sont concaténées avec les caractéristiques générées 
par toutes les couches précédentes, et le nombre de canaux est mis à jour en conséquence.
"""
##################################################################################################################################################################



def convolution_block(x, nb_channels, dropout_rate=None, bottleneck=False, weight_decay=1e-4):

    growth_rate = nb_channels/2
    # Bottleneck
    if bottleneck:
        bottleneckWidth = 4
        x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
        x = Activation('relu')(x)
        x = Conv2D(nb_channels * bottleneckWidth, (1, 1), use_bias=False, kernel_regularizer=l2(weight_decay))(x)
        # Dropout
        if dropout_rate:
            x = Dropout(dropout_rate)(x)
    
    # Standard (BN-ReLU-Conv)
    x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_channels, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(weight_decay))(x)
    
    # Dropout
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    
    return x


"""  
cette fonction implémente un bloc de convolution standard ou un bloc de bottleneck en fonction des paramètres fournis, qui est utilisé pour générer 
de nouvelles caractéristiques dans un bloc dense de l'architecture DenseNet.
"""
##################################################################################################################################################################



def transition_layer(x, nb_channels, dropout_rate=None, compression=1.0, weight_decay=1e-4):

    x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv2D(int(nb_channels*compression), (1, 1), padding='same',
                      use_bias=False, kernel_regularizer=l2(weight_decay))(x)
    
    # Adding dropout
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    return x

""" 
cette fonction implémente une couche de transition dans l'architecture DenseNet, qui réduit à la fois la dimension spatiale et le nombre de 
canaux de caractéristiques, tout en conservant les informations importantes pour le modèle.


"""

##################################################################################################################################################################



if __name__ == '__main__':
    
    model = DenseNet(input_shape = (64,64,1) , dense_blocks = 2 , dense_layers = 6 , growth_rate = 32 , nb_classes = 6 , bottleneck = True , depth = 27, weight_decay = 1e-5)
    print(model[0].summary())
    opt = SGD(lr = 0.0 , momentum = 0.9)
    model[0].compile(optimizer=opt , loss='categorical_crossentropy' , metrics=['accuracy']) 
    train_datagen = ImageDataGenerator(data_format = "channels_last")
    train_generator = train_datagen.flow_from_directory('TrainPath' , target_size = (64,64) , color_mode = 'grayscale' , batch_size = 8)  
    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    lrate = LearningRateScheduler(step_decay, verbose=1)
    callbacks_list = [lrate]
    model[0].fit(train_generator , steps_per_epoch=STEP_SIZE_TRAIN , epochs = 25, callbacks=callbacks_list, verbose=1) 
    model[0].save("SavePath")





"""  
Ce code permet de :

    - step_decay(epoch) : Cette fonction définit une stratégie de décroissance du taux d'apprentissage en fonction du numéro de l'époque lors de
    l'entraînement du modèle. Elle est utilisée comme rappel (callback) pour ajuster dynamiquement le taux d'apprentissage.


    - DenseNet(...) : Cette fonction crée un modèle DenseNet en utilisant les paramètres spécifiés tels que la forme d'entrée, le nombre de blocs denses,
    le nombre de couches dans chaque bloc dense, le taux de croissance, le nombre de classes de sortie, etc. Elle construit le modèle en utilisant des blocs 
    denses, des couches de transition et des paramètres de régularisation.


    - dense_block(x, nb_layers, nb_channels, growth_rate, dropout_rate=None, bottleneck=False, weight_decay=1e-4) : Cette fonction définit un bloc dense 
    dans le modèle DenseNet. Elle génère de nouvelles caractéristiques en concaténant les sorties de plusieurs couches de convolution.
    
    
    - convolution_block(x, nb_channels, dropout_rate=None, bottleneck=False, weight_decay=1e-4) : Cette fonction définit un bloc de convolution dans le 
    modèle DenseNet. Elle génère de nouvelles caractéristiques en appliquant des couches de convolution avec activation ReLU et optionnellement un bloc 
    bottleneck.
    
    
    - transition_layer(x, nb_channels, dropout_rate=None, compression=1.0, weight_decay=1e-4) : Cette fonction définit une couche de transition dans 
    le modèle DenseNet. Elle réduit la dimension spatiale et le nombre de canaux de caractéristiques entre les blocs denses en utilisant une combinaison 
    de convolution, normalisation et mise en commun.


    -__main__ : Cette partie du code est exécutée lorsque le script est exécuté en tant que programme principal. Elle crée, compile, entraîne et 
    sauvegarde le modèle DenseNet en utilisant les fonctions et les paramètres spécifiés précédemment.


"""