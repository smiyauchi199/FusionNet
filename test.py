import h5py
import numpy as np
import tensorflow.keras.backend as K
from tensorflow import keras
from models import Sampling, gen_mu_sigma_d, kl_loss_layer, dice_loss_sum_3d, load_3ddata, \
                   dice_loss_1, dice_loss_2, dice_loss_3, dice_loss_4, dice_loss_5, dice_loss_6, \
                   dice_loss_7, dice_loss_8, dice_loss_9, dice_loss_10
from argparse import ArgumentParser

for v in range(0,7):
    def read_datas():
        x_test = np.load('./datas/new_dataset/{}/test{}.npy'.format(v,v))
        testx = x_test[...,[0,2,4,6,8]]
        testy = x_test[...]
        print("testx shape is ", testx.shape)
        print("testy shape is ", testy.shape)
        
        def dice_mean_3d(y_true, y_pred, smooth=0.0001):#y_pred.shape -> (batch,80,80,10)
            y_true=K.cast(y_true,'float32')
            print(y_true.shape)
            print(y_pred.shape)
            union        = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
            intersection = K.sum(y_true * y_pred, axis=[1,2, 3])
            dice         = K.mean( (2. * intersection + smooth) / (union + smooth), axis=[0,1])
            return dice.numpy()
        
        def kl_loss(a, b):
            print(b)
            return b
        # loading model
        objs = {'Sampling': Sampling, 'gen_mu_sigma_d': gen_mu_sigma_d, 'kl_loss_layer': kl_loss_layer,
                'dice_loss_sum_3d': dice_loss_sum_3d, 'kl_loss': kl_loss}
        
        for i in range(1,11):exec('objs["dice_loss_%d"]=dice_loss_%d'%(i,i))
        #model = keras.models.load_model('./result/modelPreviousResearchFrame2-input9153_datanumber0{}.h5'.format(idx),custom_objects=objs)
        model = keras.models.load_model('./FusionWithoutSkipConnectionInput13579_datanumber{}.h5'.format(v),custom_objects=objs)
        # testing
        y_ = model.predict(testx)
        cls_pt = np.where(y_>0.5, 1, 0)
        DM = dice_mean_3d(testy, (y_>0.5).astype('float32'))
        print( "Mean DICE: ",DM )
        DS = dice_loss_sum_3d(testy, (y_>0.5).astype('float32')).numpy()
        print( "SUM DICE: ",DS )
        #f = h5py.File("U-NetInput13579Test_datanumber{}.hdf5".format(idx),"w")
        f = h5py.File("FusionWithoutSkipConnectionInput13579_datanumber{}.hdf5".format(v),"w")
        d1= f.create_dataset("dset1",data=cls_pt)
        f.close()

    read_datas()