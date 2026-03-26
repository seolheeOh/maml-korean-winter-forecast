import tensorflow as tf
import numpy as np
import math
import os, pathlib
from netCDF4 import Dataset
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

main_dir = '/home/doseol1129/Tool/MAML/MAML_for_climate/'
input_dir = 'dataset/'
ipath = main_dir+input_dir

shot = 5
batch_size = 8

import sys
sys.path.append(main_dir)
from Model import MAMLNets
from utils import DataLoader

exp_name = 'kortemp_DJF/train-itrNUM_TRAIN_ITER-upNUM_TRAIN_UPDATES/'
exp_path = main_dir+'output/'+exp_name+'/'

opath = exp_path+'ENS0ENSEMBLE/'
pathlib.Path(opath).mkdir(parents=True, exist_ok=True)

#  Load data
f = Dataset(ipath+'inp_nov.nc','r')
inp = f.variables['p'][:]
f.close()

zdim, ydim, xdim = inp.shape[1], inp.shape[2], inp.shape[3]

inp = inp.reshape(-1,zdim,ydim,xdim)
inp = np.array(inp, dtype=np.float32)

inp = inp/3
inp = np.swapaxes(inp,1,3)

train_inp = inp[:50]
test_inp = inp[50:]
tdim = test_inp.shape[0]

f = Dataset(ipath+'lab_DJF.nc','r')
lab = f.variables['p'][:]
f.close()

lab = lab/2
lab = np.array(lab, dtype=np.float32)

train_lab = lab[:50].reshape(-1)
test_lab = lab[50:].reshape(-1)

# Episode composition
Dataload = DataLoader(shot, xdim, ydim, zdim)

train_inp_query, train_lab_query, train_inp_support, train_lab_support = \
        Dataload.get_train_dataset(train_inp, train_lab, NUM_TRAIN_ITER, batch_size)

ts_inp_query,ts_lab_query,ts_inp_support, ts_lab_support = \
        Dataload.get_test_dataset(test_inp, test_lab, train_inp, train_lab)

print('_'*60)
print('\n Training dataset\n')
print('* Query input = ',train_inp_query.shape,' * label = ', train_lab_query.shape)
print('* Support input = ',train_inp_support.shape,' * label = ', train_lab_support.shape)

print('\n Test dataset \n')
print('* Query input = ',ts_inp_query.shape,' * label = ', ts_lab_query.shape)
print('* Support input = ',ts_inp_support.shape,' * label = ', ts_lab_support.shape)
print('_'*60)

#  Model setting
optimizer = tf.keras.optimizers.Adam()

@tf.function

def loss(x_support, y_support, x_query, y_query):
    _, loss = model(x_support, y_support, x_query, y_query)
    return loss

def train_step(x_support, y_support, x_query, y_query):
    with tf.GradientTape() as tape:
        predictions, loss = model(x_support, y_support, x_query, y_query)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(
            zip(gradients, model.trainable_variables))

    train_loss(loss)

def test_step(x_support, y_support, x_query, y_query):
    predictions, _ = model(x_support, y_support, x_query, y_query)

    return predictions

#  Model training
print('\n Training \n')

train_loss = tf.metrics.Mean()

model = MAMLNets(shot=shot, xdim=xdim, ydim=ydim, zdim=zdim,
        filter1= 4, filter2= 4, 
        update=NUM_TRAIN_UPDATES, training=True)

em = []
for itr in range (NUM_TRAIN_ITER): # Iteration

    tr_inp_support = tf.cast(train_inp_support[itr], tf.float32)
    tr_inp_query = tf.cast(train_inp_query[itr], tf.float32)

    tr_lab_support = tf.cast(train_lab_support[itr], tf.float32)
    tr_lab_query = tf.cast(train_lab_query[itr], tf.float32)
    
    train_step(tr_inp_support, tr_lab_support, tr_inp_query, tr_lab_query)
    tf.keras.backend.clear_session()
    del tr_inp_support, tr_inp_query, tr_lab_support, tr_lab_query

    template = 'iteration:{}, training loss:{:.5f}'
    print(template.format(itr+1,train_loss.result()))
            
    em.append([train_loss.result().numpy()])

model.save(opath)
del model, optimizer

train_loss = np.array(em)
tdim = len(train_loss)

oname = 'train_loss'

train_loss.astype('float32').tofile(opath+oname+'.gdat')

a = open(opath+oname+'.ctl','w')
a.write(
'dset ^'+str(oname)+'.gdat\n\
undef -9.99e+08\n\
xdef   1  linear   0.  2.5\n\
ydef   1  linear -90.  2.5\n\
zdef   1  linear 1 1\n\
tdef '+str(tdim)+'  linear jan1980 1yr\n\
vars   1\n\
p      1   1  Loss\n\
ENDVARS\n'
)
a.close()

os.system('cdo -f nc import_binary '+opath+oname+'.ctl '+opath+oname+'.nc')
os.system('rm -f '+opath+oname+'.ctl '+opath+oname+'.gdat')

#  Model test
print('\n Test \n')

tdim = len(ts_inp_query)
em_pred = []
for i in range(tdim): # meta-test-batch must be 'one point'.

    model = MAMLNets(shot=shot, xdim=xdim, ydim=ydim, zdim=zdim,
            filter1= 4, filter2= 4, 
            update=NUM_TRAIN_UPDATES, training=False)

    model.load(opath)

    test_inp_query = tf.cast(ts_inp_query[i], tf.float32)
    test_lab_query = tf.cast(ts_lab_query[i], tf.float32)

    test_inp_support = tf.cast(ts_inp_support[i], tf.float32)
    test_lab_support = tf.cast(ts_lab_support[i], tf.float32)

    predictions = test_step(test_inp_support, test_lab_support, test_inp_query, test_lab_query)

    em_pred.append(predictions)
    preds = tf.stack(em_pred)

    keras.backend.clear_session()

    del test_inp_support, test_inp_query, test_lab_support, test_lab_query

preds = preds*2
prob = np.array([preds])
prob = prob.reshape(tdim)

test_prob = np.array(prob)

#  File save
oname = 'fcst_ens0ENSEMBLE'
test_prob.astype('float32').tofile(exp_path+oname+'.gdat')

a = open(exp_path+oname+'.ctl','w')
a.write(
'dset ^'+str(oname)+'.gdat\n\
undef -9.99e+08\n\
xdef   1  linear   0.  2.5\n\
ydef   1  linear -90.  2.5\n\
zdef   1  linear 1 1\n\
tdef '+str(tdim)+'  linear jan2000 1yr\n\
vars   1\n\
p      1   1  regression\n\
ENDVARS\n'
)
a.close()

os.system('cdo -f nc import_binary '+exp_path+oname+'.ctl '+exp_path+oname+'.nc')
os.system('rm -f '+exp_path+oname+'.ctl '+exp_path+oname+'.gdat')

