
import csv
import re
import numpy as np

#    https://keras.io/getting-started/sequential-model-guide/#examples
#  check this out
#  check this out for shaping input and outputs#  check this out for shaping input and outputs#  check this out for shaping input and outputs#  check this out for shaping input and outputs
#  check this out for shaping input and outputs#  check this out for shaping input and outputs#  check this out for shaping input and outputs#  check this out for shaping input and outputs#  check this out for shaping input and outputs
#  check this out for shaping input and outputs#  check this out for shaping input and outputs#  check this out for shaping input and outputs#  check this out for shaping input and outputs

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, TimeDistributed
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.preprocessing import sequence 


batch_size = 1000
#symbol_list = ['0_1_1' '2_1_1' '7_1_1' 'c_1_1']  ==> acc=0.95 by 60 epoch

symbol_list = ['0_1_1', '1_1_1', '2_1_1', '3_1_1',  '4_1_1', '4_2_nose',  
               '5_1_1',  '5_2_hook', '6_1_1',  '7_1_1', '8_1_1', '9_1_1', 
               'a_1_1', 'b_1_1', 'c_1_1', 'd_1_1',  ]
#file_ = './abc_head_essence_final_2013.csv'
file_ = './abc_essence_final_2013.csv'
t = [] 
k = []
l = []
s = []
verbose = False
with open(file_, 'r') as f:
    #reader = csv.reader(f, delimiter=',', quotechar = '"', doublequote = True, quoting=csv.QUOTE_NONE)
    reader = csv.reader(f, skipinitialspace=True) # delimiter=',', quotechar = '"', doublequote = True, quoting=csv.QUOTE_NONE)
    next(reader, None)  #skipping header
    for row in reader:
        trace_regular =  row[13]
        key =  row[-1]

        #https://www.regular-expressions.info/floatingpoint.html
        refindout = re.findall(r"[-+]?[0-9]*\.?[0-9]+", trace_regular)
        map_float = np.array( list( map(float, refindout)))
        strokes = np.reshape( map_float , (-1, 2))
        if( key in  symbol_list):
          t.append(strokes)
          k.append(key)
          l.append(len(strokes))
          s.append(row[0]) #sequence

        if(verbose):
          print(row)
          print(trace_regular)
          print(refindout)
          print( map_float)


t = np.asarray(t)

datasize = len(l)
print( " read %d strokes from %s " % (datasize, file_ ))
#print(l)

def TRACE_print( t ):
  for trace in t:
     print( "len:%d, first %s %s, last %s %s" % (len(trace), trace[0,0], trace[0,1], trace[-1,0],trace[-1,1]))
     print( trace.shape )
     
def zeropad ( t, timestep): # t is batch size of (N, 2) , N is number of points
  batch = len(t)
  #print( " batch %d , timestep %d " % (batch, timestep))
  #print(timestep)
  new_x = np.zeros((batch, timestep, 2))
  for i in range(0,batch):
     #print( len(t[i,]) )
     new_x[i,:len(t[i,])] = t[i,]
     #print(t[i,])
     #print(new_x[i,])
  return(new_x)

def findindex ( lib, keys ):
  idex = []
  library = lib.tolist()
  for k in keys:
    #print(k)
    idex.append(library.index(k))
  #print(idex)
  return(idex)


#sequence_length = 100'
#k_list = np.sort(k)
#print(k_list)
k_set = np.sort(list(set( k )))


#idx = findindex( k_set, list(["a_1_1", "b_1_1", "z_1_1"]))
#print(idx)
#k_list = list(k_set)
print( k_set )
k_count = len(k_set)
print(k_count)
current = 0
#mask = np.array([[1,   1], [1,   1],[1,   1],[1,   1],[1,   1],  [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
#print (mask)
def train_generator():
  global current
  global datasize
  global t, k, l, s
  global verbose
  global k_set, k_count
  while True:

        begin = current
        end = begin + batch_size
        if( end > datasize): 
            end = end - datasize
            x_train = np.append(t[begin:], t[:end])
            y_train = np.append(k[begin:], k[:end]) #k[begin:] + k[:end]
            l_train = np.append(l[begin:], l[:end])
            s_train = np.append(s[begin:], s[:end])            
        else:
            x_train = t[begin:end]
            y_train = k[begin:end]
            l_train = l[begin:end]
            s_train = s[begin:end]        
        

        current = end
        timestep = int( max(l_train) )

        x_train = zeropad(x_train, timestep) # np.zeros((timestep, 2))
        idx = findindex( k_set,  y_train)
        category_y = to_categorical(idx , num_classes = k_count)
        #TRACE_print( x_train )
        #print( "x_shape ")
        if(verbose):
          print( " begin %d , end %d " % (begin, end))
          print(x_train.shape)
          print( y_train )
          print(category_y)
 
        #print( l_train )
        if(verbose):
          print( s_train )
        #print( "max_time %d" % (timestep) )

##        if(i == 500*100):
##           print(i)
          print(x_train.shape)
##           print(x_train[i,])
##           print(y_train[i])

        yield x_train, category_y

#t_g = train_generator()
#for i in range(0,10):
#    print( " ------------------------------- %d"% i )
#    next(t_g)



model = Sequential()
#model.add(LSTM(32, return_sequences=True, input_shape=(None, 2)))
#model.add(LSTM(8, return_sequences=False)) 

model.add(LSTM(64, return_sequences=True, input_shape=(None, 2)))
model.add(LSTM(16, return_sequences=False)) 

#model.add((Dense(3, activation='sigmoid')))  
model.add((Dense(k_count, activation='softmax')))  

#The sigmoid activation is outputing values between 0 and 1 independently from one another.
#If you want probabilities outputs that sums up to 1, use the softmax activation on your last layer, it will normalize the output to sum up to 1. 

#model.add((Dense(2, activation='softmax')))  

	#timedistributed requires all sequences (return_sequences = True). 
	#https://stackoverflow.com/questions/43034960/many-to-one-and-many-to-many-lstm-examples-in-keras

print(model.summary(90))

model.compile(loss='categorical_crossentropy',
                optimizer='adam',  metrics=['mse', 'accuracy'])
#    https://github.com/keras-team/keras/issues/2548
#    evaluate() will return the list of metrics that the model was compiled with. 
#    So if your compile metrics list include 'accuracy', then this should still work.

 
#model.fit_generator(train_generator(), steps_per_epoch=30, epochs=10, verbose=1)
num_batch_in_epoch = round(datasize / batch_size) + 1
#model.fit_generator(train_generator(), steps_per_epoch=num_batch_in_epoch, epochs=30, verbose=1)
model.fit_generator(train_generator(), steps_per_epoch=30, epochs=90, verbose=1)


model.save('./math_equation.h5')


# #Test Yo!!!!!
# #Test Yo!!!!!
# #Test Yo!!!!!
# #Test Yo!!!!!
# #Test Yo!!!!!
# #Test Yo!!!!!
# import numpy as np
# from tensorflow.python.keras.models import load_model
# model = load_model('/home/young/Tensorflow_projects/coding/keras_rnn_many_to_one_X.h5')
# #the batch size of 1 test sample
# x_test_up = np.array([[0,     0.1], [0.1,     0.3], [0.2,     0.5], [0.3, 0.7], [0.4,     0.9], [0.5,     1.1], [0.6,     1.3], [0.7, 1.5], [0.8,     1.7], [0.9,     1.9]]) 
# x_test_up = x_test_up.reshape(1,10,2)
# print( model.predict(x_test_up, batch_size=None, verbose=1) )
# #   array([[0.00663349, 0.6025158 ]], dtype=float32)   -> 0/1 , 1 is greater -> up sloping pattern

# x_test_decreasing  = np.array([[0,   1], [0.1,   0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, .5], [0.6,  .4], [0.7,  .3], [0.8,  .2], [0.9,  .1]]) 
# x_test_decreasing= x_test_decreasing.reshape(1,10,2)
# model.predict(x_test_decreasing, batch_size=None, verbose=1)
# #   array([[0.5322375 , 0.00834433]], dtype=float32)   -> 0/1 , 0 is greater -> ----> correctly identifying it is decreasing!!!!

# x_test_down_up  = np.array([[0,   1], [0.1,   0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, .6], [0.6,  .7], [0.7,  .8], [0.8,  .9], [0.9,  1]]) 
# x_test_down_up= x_test_down_up.reshape(1,10,2)
# model.predict(x_test_down_up, batch_size=None, verbose=1)
# #   array([[0.5102862 , 0.00811925]], dtype=float32)   -> 0/1 , 0 is greater ->-----> down and then up --- recognized as down

# x_test_up_down  = np.array([[0,     0.1], [0.1,     0.3], [0.2,     0.5], [0.3, 0.7], [0.4,     0.9], [0.5,     .6], [0.6,     .5], [0.7, .4], [0.8,     .3], [0.9,     .2]]) 
# x_test_up_down= x_test_up_down.reshape(1,10,2)
# model.predict(x_test_up_down, batch_size=None, verbose=1)
# #   array([[0.3975516, 0.011379 ]], dtype=float32)     -> 0/1 , 0 is greater ->---> this also down? 

# x_test_up_flat = np.array([[0,     0.1], [0.1,     0.3], [0.2,     0.5], [0.3, 0.7], [0.4,     0.9], [0.5,     0.9], [0.6,     0.9], [0.7, 0.9], [0.8,     0.9], [0.9,     0.9]]) 
# x_test_up_flat = x_test_up_flat.reshape(1,10,2)
# model.predict(x_test_up_flat, batch_size=None, verbose=1)
# #   array([[0.0125859 , 0.42021653]], dtype=float32)   -> 0/1 , 1 is greater -> recognized as up sloping
 




# Epoch 1/10
# 30/30 [==============================]30/30 [==============================] - 3s 103ms/step - loss: 1.0252 - mean_squared_error: 0.2066 - acc: 0.4854

# Epoch 2/10
# 30/30 [==============================]30/30 [==============================] - 2s 76ms/step - loss: 0.8104 - mean_squared_error: 0.1600 - acc: 0.7046

# Epoch 3/10
# 30/30 [==============================]30/30 [==============================] - 2s 75ms/step - loss: 0.5686 - mean_squared_error: 0.1050 - acc: 0.8064

# Epoch 4/10
# 30/30 [==============================]30/30 [==============================] - 2s 77ms/step - loss: 0.2436 - mean_squared_error: 0.0361 - acc: 0.9540

# Epoch 5/10
# 30/30 [==============================]30/30 [==============================] - 2s 75ms/step - loss: 0.0628 - mean_squared_error: 0.0055 - acc: 0.9936

# Epoch 6/10
# 30/30 [==============================]30/30 [==============================] - 2s 77ms/step - loss: 0.0356 - mean_squared_error: 0.0024 - acc: 0.9967

# Epoch 7/10
# 30/30 [==============================]30/30 [==============================] - 2s 77ms/step - loss: 0.0251 - mean_squared_error: 0.0014 - acc: 0.9980

# Epoch 8/10
# 30/30 [==============================]30/30 [==============================] - 2s 76ms/step - loss: 0.0208 - mean_squared_error: 0.0012 - acc: 0.9981

# Epoch 9/10
# 30/30 [==============================]30/30 [==============================] - 2s 76ms/step - loss: 0.0176 - mean_squared_error: 9.7123e-04 - acc: 0.9983

# Epoch 10/10
# 30/30 [==============================]30/30 [==============================] - 2s 77ms/step - loss: 0.0158 - mean_squared_error: 9.8142e-04 - acc: 0.9982

# $ ^C
# $ 


# 244
#  7/30 [======>.......................] 7/30 [======>.......................] - ETA: 16s - loss: 0.5758 - mean_squared_error: 0.0804 - acc: 0.79311000
# 249
#  8/30 [=======>......................] 8/30 [=======>......................] - ETA: 17s - loss: 0.5600 - mean_squared_error: 0.0776 - acc: 0.80391000
# 143
#  9/30 [========>.....................] 9/30 [========>.....................] - ETA: 17s - loss: 0.5596 - mean_squared_error: 0.0776 - acc: 0.80571000
# 153.0
# 10/30 [=========>....................]10/30 [=========>....................] - ETA: 16s - loss: 0.5524 - mean_squared_error: 0.0763 - acc: 0.8096Traceback (most recent call last):
#   File "train.keras.math.eq.py", line 179, in <module>
#     model.fit_generator(train_generator(), steps_per_epoch=30, epochs=30, verbose=1)
#   File "/home/young/anaconda3/lib/python3.6/site-packages/tensorflow/python/keras/_impl/keras/models.py", line 1198, in fit_generator
#     initial_epoch=initial_epoch)
#   File "/home/young/anaconda3/lib/python3.6/site-packages/tensorflow/python/keras/_impl/keras/engine/training.py", line 2143, in fit_generator
#     generator_output = next(output_generator)
#   File "/home/young/anaconda3/lib/python3.6/site-packages/tensorflow/python/keras/_impl/keras/utils/data_utils.py", line 744, in get
#     six.reraise(value.__class__, value, value.__traceback__)
#   File "/home/young/anaconda3/lib/python3.6/site-packages/six.py", line 693, in reraise
#     raise value
#   File "/home/young/anaconda3/lib/python3.6/site-packages/tensorflow/python/keras/_impl/keras/utils/data_utils.py", line 644, in data_generator_task
#     generator_output = next(self._generator)
#   File "train.keras.math.eq.py", line 125, in train_generator
#     x_train = zeropad(x_train, timestep) # np.zeros((timestep, 2))
#   File "train.keras.math.eq.py", line 67, in zeropad
#     new_x = np.zeros((batch, timestep, 2))
# TypeError: 'numpy.float64' object cannot be interpreted as an integer



  #if( len(t) > 0):
