import keras.backend as K
from keras.engine import Input,Model
import keras
from keras.optimizers import Adam
from keras.layers import BatchNormalization,Activation,Conv2D,MaxPooling2D,Conv2DTranspose,UpSampling2D
import metrics as m
from keras.layers.core import Lambda
import numpy as np


def up_and_concate(down_layer, layer):
    in_channel = down_layer.get_shape().as_list()[3]
    out_channel = in_channel // 2
    up = Conv2DTranspose(out_channel,[2,2],strides=[2,2])(down_layer)
    print("--------------")
    print(str(up.get_shape()))
    print(str(layer.get_shape()))
    print("--------------")
    my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))
    concate = my_concat([up, layer])
    # must use lambda 
    #concate=K.concatenate([up, layer], 3)
    return concate


def attention_block_2d(x, g, inter_channel):
    '''

    :param x: x input from down_sampling same layer output (?,x_height,x_width,x_channel)
    :param g: gate input from up_sampling layer last output (?,g_height,g_width,g_channel)
    g_height,g_width=x_height/2,x_width/2
    :return:
    '''
    print('attention_block:')
    # theta_x(?,g_height,g_width,inter_channel)
    theta_x = Conv2D(inter_channel, [2, 2], strides=[2, 2])(x)
    print(str(theta_x.get_shape()))
    # phi_g(?,g_height,g_width,inter_channel)
    phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1])(g)
    print(str(phi_g.get_shape()))
    # f(?,g_height,g_width,inter_channel)
    f = Activation('relu')(keras.layers.add([theta_x, phi_g]))
    print(str(f.get_shape()))
    # psi_f(?,g_height,g_width,1)
    psi_f = Conv2D(1, [1, 1], strides=[1, 1])(f)
    print(str(psi_f.get_shape()))
    # sigm_psi_f(?,g_height,g_width)
    sigm_psi_f = Activation('sigmoid')(psi_f)
    print(str(sigm_psi_f.get_shape()))
    # rate(?,x_height,x_width)
    rate = UpSampling2D(size=[2, 2])(sigm_psi_f)
    print(str(rate.get_shape()))
    # att_x(?,x_height,x_width,x_channel)
    att_x = keras.layers.multiply([x, rate])
    print(str(att_x.get_shape()))
    print('-----------------')
    return att_x


def unet_model_2d_attention(input_shape,n_labels,batch_normalization=False,initial_learning_rate=0.00001,metrics=m.MIoU):
    """
    input_shape:without batch_size,(img_height,img_width,img_depth)
    metrics:
    """


    inputs=Input(input_shape)

    down_layer=[]

    layer=inputs

    #down_layer_1
    layer=res_block_v2(layer,64,batch_normalization=batch_normalization)
    down_layer.append(layer)
    layer=MaxPooling2D(pool_size=[2,2],strides=[2,2])(layer)

    print(str(layer.get_shape()))

    # down_layer_2
    layer = res_block_v2(layer, 128, batch_normalization=batch_normalization)
    down_layer.append(layer)
    layer = MaxPooling2D(pool_size=[2, 2], strides=[2, 2])(layer)

    print(str(layer.get_shape()))

    # down_layer_3
    layer = res_block_v2(layer, 256, batch_normalization=batch_normalization)
    down_layer.append(layer)
    layer = MaxPooling2D(pool_size=[2, 2], strides=[2, 2])(layer)

    print(str(layer.get_shape()))

    # down_layer_4
    layer = res_block_v2(layer, 512, batch_normalization=batch_normalization)
    down_layer.append(layer)
    layer = MaxPooling2D(pool_size=[2, 2], strides=[2, 2])(layer)

    print(str(layer.get_shape()))

    # bottle_layer
    layer = res_block_v2(layer, 1024, batch_normalization=batch_normalization)
    print(str(layer.get_shape()))

    # up_layer_4
    layer = attention_block_2d( down_layer[3],layer,256)
    layer = res_block_v2(layer, 512,batch_normalization=batch_normalization)
    print(str(layer.get_shape()))

    # up_layer_3
    layer =  attention_block_2d( down_layer[2],layer,128)
    layer = res_block_v2(layer, 256, batch_normalization=batch_normalization)
    print(str(layer.get_shape()))

    # up_layer_2
    layer =  attention_block_2d( down_layer[1],layer,64)
    layer = res_block_v2(layer, 128, batch_normalization=batch_normalization)
    print(str(layer.get_shape()))

    # up_layer_1
    layer =  attention_block_2d( down_layer[0],layer,32)
    layer = res_block_v2(layer, 64, batch_normalization=batch_normalization)
    print(str(layer.get_shape()))

    # score_layer
    layer = Conv2D(n_labels,[1,1],strides=[1,1])(layer)
    print(str(layer.get_shape()))

    # softmax
    layer = Activation('softmax')(layer)
    print(str(layer.get_shape()))

    outputs=layer

    model=Model(inputs=inputs,outputs=outputs)

    metrics=[metrics]

    model.compile(optimizer=Adam(lr=initial_learning_rate), loss=m.MIoU_loss, metrics=metrics)


    return model


def unet_model_2d(input_shape,n_labels,batch_normalization=False,initial_learning_rate=0.00001,metrics=m.MIoU):
    """
    input_shape:without batch_size,(img_height,img_width,img_depth)
    metrics:
    """


    inputs=Input(input_shape)

    down_layer=[]

    layer=inputs

    #down_layer_1
    layer=res_block_v2(layer,64,batch_normalization=batch_normalization)
    down_layer.append(layer)
    layer=MaxPooling2D(pool_size=[2,2],strides=[2,2])(layer)

    print(str(layer.get_shape()))

    # down_layer_2
    layer = res_block_v2(layer, 128, batch_normalization=batch_normalization)
    down_layer.append(layer)
    layer = MaxPooling2D(pool_size=[2, 2], strides=[2, 2])(layer)

    print(str(layer.get_shape()))

    # down_layer_3
    layer = res_block_v2(layer, 256, batch_normalization=batch_normalization)
    down_layer.append(layer)
    layer = MaxPooling2D(pool_size=[2, 2], strides=[2, 2])(layer)

    print(str(layer.get_shape()))

    # down_layer_4
    layer = res_block_v2(layer, 512, batch_normalization=batch_normalization)
    down_layer.append(layer)
    layer = MaxPooling2D(pool_size=[2, 2], strides=[2, 2])(layer)

    print(str(layer.get_shape()))

    # bottle_layer
    layer = res_block_v2(layer, 1024, batch_normalization=batch_normalization)
    print(str(layer.get_shape()))

    # up_layer_4
    layer = up_and_concate(layer, down_layer[3])
    layer = res_block_v2(layer, 512,batch_normalization=batch_normalization)
    print(str(layer.get_shape()))

    # up_layer_3
    layer = up_and_concate(layer, down_layer[2])
    layer = res_block_v2(layer, 256, batch_normalization=batch_normalization)
    print(str(layer.get_shape()))

    # up_layer_2
    layer = up_and_concate(layer, down_layer[1])
    layer = res_block_v2(layer, 128, batch_normalization=batch_normalization)
    print(str(layer.get_shape()))

    # up_layer_1
    layer = up_and_concate(layer, down_layer[0])
    layer = res_block_v2(layer, 64, batch_normalization=batch_normalization)
    print(str(layer.get_shape()))

    # score_layer
    layer = Conv2D(n_labels,[1,1],strides=[1,1])(layer)
    print(str(layer.get_shape()))

    # softmax
    layer = Activation('softmax')(layer)
    print(str(layer.get_shape()))

    outputs=layer

    model=Model(inputs=inputs,outputs=outputs)

    metrics=[metrics]

    model.compile(optimizer=Adam(lr=initial_learning_rate), loss=m.MIoU_loss, metrics=metrics)


    return model



def res_block_v2(input_layer,out_n_filters,batch_normalization=False,kernel_size=[3,3],stride=[1,1],padding='same'):

    input_n_filters = input_layer.get_shape().as_list()[3]
    print(str(input_layer.get_shape()))
    layer=input_layer

    for i in range(2):
        if batch_normalization:
            layer=BatchNormalization()(layer)
        layer=Activation('relu')(layer)
        layer=Conv2D(out_n_filters,kernel_size,strides=stride,padding=padding)(layer)

    if out_n_filters!=input_n_filters:
        skip_layer=Conv2D(out_n_filters,[1,1],strides=stride,padding=padding)(input_layer)
    else:
        skip_layer=input_layer

    out_layer=keras.layers.add([layer,skip_layer])


    return out_layer




