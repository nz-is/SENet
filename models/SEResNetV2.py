import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D,\
    ZeroPadding2D, Dense, multiply, Reshape, Conv2D, \
    Flatten, add, BatchNormalization, Activation, Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from keras.regularizers import l2
from utils import se_block

__all__ = ["SEResNet50", "SEResNet18", "SEResNet50", "SEResNet101", "SEResNet154"]

class SEResNet:  
  @staticmethod
  def residual_module(in_block, K, stride, chanDim, red=False,
                        reg= 1e-4, bnEps=2e-5, bnMom=.9):
      """Creates Pre-activation bottleneck residual module + 
      SE block
     
      Arguments:
        in_block: input keras tensor
        K: no. of channels out
        red: boolean flag for reduction in spatial feature map
        reg: regulaizer param for conv.

      Returns: a Keras tensor 

      References:
        -   [ResNet](https://arxiv.org/abs/1512.03385)
        -   [Squeeze-and-Excitation Networks ] (https://arxiv.org/abs/1709.01507)
      """

      x = in_block
      chan_in = in_block.shape.as_list()[-1]
      bn1 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
      relu1 = Activation("relu")(bn1)
      conv1 = Conv2D(filters=int(K * .25), kernel_size=(1, 1), 
                     use_bias=False,
                     kernel_regularizer=l2(reg))(relu1) #Conv2D learns 1/4(0.25) of the last conv filter

      bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv1)
      relu2 = Activation("relu")(bn2)
      conv2 = Conv2D(filters=int(K * .25), kernel_size=(3, 3), strides=stride, 
                     padding="same",
                     use_bias=False,
                     kernel_regularizer=l2(reg))(relu2) #Conv2D learns 1/4(0.25) of the last conv filter

      bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv2)
      relu3 = Activation("relu")(bn3)
      conv3 = Conv2D(filters=K, kernel_size=(1, 1),  
                     kernel_regularizer=l2(reg))(relu3) 

      if red:
        print("Stride at conv res: {}".format(stride))
        x = Conv2D(filters=K, kernel_size=(1, 1), strides=stride,
                    kernel_regularizer=l2(reg))(relu1)
      x = se_block(x)
      print("Residual: {}, conv3: {}".format(x.shape, conv3.shape))
      return add([x, conv3])

  @staticmethod
  def build(width, height, depth, classes, stages, filters, include_top, pooling,
            reg=1e-3, bnEps=2e-5, bnMom=0.0):
      """ Instantiate the Squeeze and Excite ResNet architecture. Note that ,
        when using TensorFlow for best performance you should set
        `image_data_format="channels_last"` in your Keras config
        at ~/.keras/keras.json.
        The dimension ordering
        convention used by the model is the one
        specified in your Keras config file.
        # Arguments
            initial_conv_filters: number of features for the initial convolution
            depth: number or layers in the each block, defined as a list.
                ResNet-50  = [3, 4, 6, 3]
                ResNet-101 = [3, 6, 23, 3]
                ResNet-152 = [3, 8, 36, 3]
            filters: number of filters per block, defined as a list.
                filters = [64, 128, 256, 512
            reg: weight decay (l2 norm)
            include_top: whether to include the fully-connected
                layer at the top of the network.
            input_shape: optional shape tuple, only to be specified
                if `include_top` is False (otherwise the input shape
                has to be `(224, 224, 3)` (with `tf` dim ordering)
                or `(3, 224, 224)` (with `th` dim ordering).
                It should have exactly 3 inputs channels,
                and width and height should be no smaller than 8.
                E.g. `(200, 200, 3)` would be one valid value.
            pooling: Optional pooling mode for feature extraction
                when `include_top` is `False`.
                - `None` means that the output of the model will be
                    the 4D tensor output of the
                    last convolutional layer.
                - `avg` means that global average pooling
                    will be applied to the output of the
                    last convolutional layer, and thus
                    the output of the model will be a 2D tensor.
                - `max` means that global max pooling will
                    be applied.
            classes: optional number of classes to classify images
                into, only to be specified if `include_top` is True, and
                if no `weights` argument is specified.

        # Returns
            A Keras model instance.
      """
      inputShape = (height, width, depth)
      chanDim = -1

      if K.image_data_format() == "channels_first": 
        inputShape = (depth, height, width) 
        chanDim = 1

      inputs = Input(shape=inputShape)
   

      # block 1 (initial conv block)
      x = ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(inputs)
      x = Conv2D(64, (7,7), use_bias=False, strides=(2,2), 
                  kernel_initializer="he_normal", kernel_regularizer=l2(reg))(x)
      x = BatchNormalization(axis=chanDim, name="bn_conv1")(x)
      x = Activation("relu")(x)
      x = ZeroPadding2D(padding=((1,1), (1,1)), name="pool1_pad")(x)
      x = MaxPooling2D(3, strides=2)(x)

      for i in range(0, len(stages)):
        stride = (1,1) if i == 0 else (2,2) # block 2 (projection block) w stride(1,1)

        print("Stage {}, Stride={}".format(i, stride))
        x = SEResNet.residual_module(x, filters[i+1], stride, 
                                  chanDim=chanDim, red=True, bnEps=bnEps, bnMom=bnMom)
        for j in range(0, stages[i] + 1): #stacking res block to each depth layer
          x = SEResNet.residual_module(x, filters[i+1], stride=(1,1),
                                    chanDim=chanDim, bnEps=bnEps, 
                                     bnMom=bnMom)
      x = BatchNormalization(axis=chanDim, epsilon=bnEps, 
                              momentum=bnMom)(x)
      x = Activation("relu")(x)

      if include_top:
        x = GlobalAveragePooling2D()(x)
        x = Dense(classes, use_bias=False, kernel_regularizer=l2(reg),
                  activation='softmax')(x)
      else:
          if pooling == 'avg':
              print("Adding average pool")
              x = GlobalAveragePooling2D()(x)
          elif pooling == 'max':
              x = GlobalMaxPooling2D()(x)
     
      model = Model(inputs=inputs, outputs=x, name="SEResNet")
      return model 

  def SEResNet18(input_shape, 
               classes=1000, 
               include_top=None, 
               input_tensor=None, 
               pooling=None):
  
    (height, width, depth) = (input_shape[1], input_shape[2], input_shape[3])

    return SEResNet.build(width=width, height=height, depth=depth, classes=classes, stages=(3, 3, 3, 3), 
                      include_top=include_top, pooling=pooling,
                        filters=(64, 256, 512, 1024, 2048), reg=1e-5, bnEps=2e-5, bnMom=0.0)
  
  def SEResNet50(input_shape, 
               classes=1000, 
               include_top=None, 
               input_tensor=None, 
               pooling=None):
  
    (height, width, depth) = (input_shape[1], input_shape[2], input_shape[3])

    return SEResNet.build(width=width, height=height, depth=depth, classes=classes, stages=(3, 4, 6, 3), 
                      include_top=include_top, pooling=pooling,
                        filters=(64, 256, 512, 1024, 2048), reg=1e-5, bnEps=2e-5, bnMom=0.0)
  def SEResNet101(input_shape, 
               classes=1000, 
               include_top=None, 
               input_tensor=None, 
               pooling=None):
  
    (height, width, depth) = (input_shape[1], input_shape[2], input_shape[3])

    return SEResNet.build(width=width, height=height, depth=depth, classes=classes, stages=(3, 6, 23, 3), 
                      include_top=include_top, pooling=pooling,
                        filters=(64, 256, 512, 1024, 2048), reg=1e-5, bnEps=2e-5, bnMom=0.0)
 
  def SEResNet152(input_shape, 
               classes=1000, 
               include_top=None, 
               input_tensor=None, 
               pooling=None):
  
    (height, width, depth) = (input_shape[1], input_shape[2], input_shape[3])

    return SEResNet.build(width=width, height=height, depth=depth, classes=classes, 
                          stages=[3, 8, 36, 3], 
                      include_top=include_top, pooling=pooling,
                        filters=(64, 256, 512, 1024, 2048), reg=1e-5, bnEps=2e-5, bnMom=0.0)
