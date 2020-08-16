from tensorflow.keras.layers import GlobalAveragePooling2D,\
   Dense, multiply, Reshape
from tensorflow.keras import backend as K
   
def se_block(in_block, ratio=16):
  """Creates channel-wise squeeze and excite block 

  args:
    input_tensor: input keras tensor
    ch: no. of channels in 
    ratio: no. of filters out

  Returns: a Keras tensor 

  References:
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
  """
  channel_axis = 1 if K.image_data_format() == "channels_first" else -1
  filters = in_block.shape.as_list()[channel_axis]
  x = GlobalAveragePooling2D()(in_block) # a vector of N dim. of in_block
  x = Reshape((1, 1, filters))(x)
  x = Dense(filters//ratio, activation="relu")(x)
  x = Dense(filters, activation="sigmoid")(x)

  return multiply([in_block, x])