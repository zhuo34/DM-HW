import numpy as np

def relu_feedforward(in_):
    '''
    The feedward process of relu
      in_:
              in_	: the input, shape: any shape of matrix
      
      outputs:
              out : the output, shape: same as in
    '''
    out = np.zeros(in_.shape)
    out[in_ > 0] = in_[in_ > 0]
    return out
