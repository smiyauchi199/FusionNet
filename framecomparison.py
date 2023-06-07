
import tensorflow.keras.backend as K
import tensorflow as tf
import h5py
import numpy as np

for i in range(7):
    def load_3ddata():
        Path_to_predDatas = "./Fusion13579KL_datanumber{}.hdf5".format(i)
        #Path_to_gtDatas = "./datas/new_dataset/test{}.h5".format(i)
        pred_File = h5py.File(Path_to_predDatas, 'r')
        gt_File = np.load('./datas/new_dataset/test{}.npy'.format(i))
        # gt_File = h5py.File(Path_to_gtDatas, 'r')

        #tr_Data = tr_File
        pred = pred_File['dset1']
        #gt = gt_File['train'][start:end,...]

        # to close the file
        # pred.close()
        # gt.close()
        return pred, gt_File

    pred, gt_File = load_3ddata()
    print(pred.shape)
    print(gt_File.shape)

    def dice_loss_per_frame(y_true, y_pred, smooth=0.0001,index=0):#y_pred.shape -> (batch,80,80,10)
        y_true=K.cast(y_true,'float32')[...,index]
        # y_pred = K.cast(y_pred, [10, 80, 80, 80, 10])
        y_pred=K.cast(y_pred,'float32')[...,index]
        union        = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
        intersection = K.sum(y_true * y_pred, axis=[1,2,3])
        dice         = K.mean( (2. * intersection + smooth) / (union + smooth), axis=[0])
        return dice
    def dice_loss_1(y_true, y_pred,index=0):return dice_loss_per_frame(y_true=y_true, y_pred=y_pred,index=index)
    def dice_loss_2(y_true, y_pred,index=1):return dice_loss_per_frame(y_true=y_true, y_pred=y_pred,index=index)
    def dice_loss_3(y_true, y_pred,index=2):return dice_loss_per_frame(y_true=y_true, y_pred=y_pred,index=index)
    def dice_loss_4(y_true, y_pred,index=3):return dice_loss_per_frame(y_true=y_true, y_pred=y_pred,index=index)
    def dice_loss_5(y_true, y_pred,index=4):return dice_loss_per_frame(y_true=y_true, y_pred=y_pred,index=index)
    def dice_loss_6(y_true, y_pred,index=5):return dice_loss_per_frame(y_true=y_true, y_pred=y_pred,index=index)
    def dice_loss_7(y_true, y_pred,index=6):return dice_loss_per_frame(y_true=y_true, y_pred=y_pred,index=index)
    def dice_loss_8(y_true, y_pred,index=7):return dice_loss_per_frame(y_true=y_true, y_pred=y_pred,index=index)
    def dice_loss_9(y_true, y_pred,index=8):return dice_loss_per_frame(y_true=y_true, y_pred=y_pred,index=index)
    def dice_loss_10(y_true, y_pred,index=9):return dice_loss_per_frame(y_true=y_true, y_pred=y_pred,index=index)

    print(dice_loss_1(gt_File, pred, index=0))
    print(dice_loss_2(gt_File, pred, index=1))
    print(dice_loss_3(gt_File, pred, index=2))
    print(dice_loss_4(gt_File, pred, index=3))
    print(dice_loss_5(gt_File, pred, index=4))
    print(dice_loss_6(gt_File, pred, index=5))
    print(dice_loss_7(gt_File, pred, index=6))
    print(dice_loss_8(gt_File, pred, index=7))
    print(dice_loss_9(gt_File, pred, index=8))
    print(dice_loss_10(gt_File, pred, index=9))
# for i in range(7):
#     def load_3ddata():
#         Path_to_predDatas = "./newwwresult/Input13579DataAugmentation/modelPreviousResearchFrame4-input1375_datanumber{}.hdf5".format(i)
#         #Path_to_gtDatas = "./datas/new_dataset/test{}.h5".format(i)
#         pred_File = h5py.File(Path_to_predDatas, 'r')
#         gt_File = np.load('./datas/new_dataset/test{}.npy'.format(i))
#         # gt_File = h5py.File(Path_to_gtDatas, 'r')

#         #tr_Data = tr_File
#         pred = pred_File['dset1']
#         #gt = gt_File['train'][start:end,...]

#         # to close the file
#         # pred.close()
#         # gt.close()
#         return pred, gt_File

#     pred, gt_File = load_3ddata()

#     def dice_loss_per_frame(y_true, y_pred, smooth=0.0001,index=0):#y_pred.shape -> (batch,80,80,10)
#         y_true=K.cast(y_true,'float32')[...,index+3]
#         # y_pred = K.cast(y_pred, [10, 80, 80, 80, 10])
#         y_pred=K.cast(y_pred,'float32')[...,index]
#         union        = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
#         intersection = K.sum(y_true * y_pred, axis=[1,2,3])
#         dice         = K.mean( (2. * intersection + smooth) / (union + smooth), axis=[0])
#         return dice

#     def dice_loss_1(y_true, y_pred,index=0):return dice_loss_per_frame(y_true=y_true, y_pred=y_pred,index=index)
#     def dice_loss_2(y_true, y_pred,index=1):return dice_loss_per_frame(y_true=y_true, y_pred=y_pred,index=index)
#     def dice_loss_3(y_true, y_pred,index=2):return dice_loss_per_frame(y_true=y_true, y_pred=y_pred,index=index)
#     def dice_loss_4(y_true, y_pred,index=3):return dice_loss_per_frame(y_true=y_true, y_pred=y_pred,index=index)
#     def dice_loss_5(y_true, y_pred,index=4):return dice_loss_per_frame(y_true=y_true, y_pred=y_pred,index=index)
#     def dice_loss_6(y_true, y_pred,index=5):return dice_loss_per_frame(y_true=y_true, y_pred=y_pred,index=index)
#     def dice_loss_7(y_true, y_pred,index=6):return dice_loss_per_frame(y_true=y_true, y_pred=y_pred,index=index)
#     def dice_loss_8(y_true, y_pred,index=7):return dice_loss_per_frame(y_true=y_true, y_pred=y_pred,index=index)
#     def dice_loss_9(y_true, y_pred,index=8):return dice_loss_per_frame(y_true=y_true, y_pred=y_pred,index=index)
#     def dice_loss_10(y_true, y_pred,index=9):return dice_loss_per_frame(y_true=y_true, y_pred=y_pred,index=index)

#     print(dice_loss_1(gt_File, pred, index=0))

