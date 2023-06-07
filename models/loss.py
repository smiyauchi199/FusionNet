import tensorflow.keras.backend as K
import tensorflow as tf
def dice_loss_per_frame(y_true, y_pred, smooth=0.0001,index=0):#y_pred.shape -> (batch,80,80,10)
    y_true=K.cast(y_true,'float32')[...,index]
    y_pred = tf.reshape(y_pred, [10, 80, 80, 80, 10])
    y_pred=y_pred[...,index]
    union        = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    dice         = K.mean( (2. * intersection + smooth) / (union + smooth), axis=[0])
    return 1-dice

def dice_loss_sum_3d(y_true, y_pred, smooth=0.0001):
    y_true=K.cast(y_true,'float32')
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    dice_loss    = K.sum(1 - (2. * intersection + smooth) / (union + smooth), axis=[0,1]) # dice < batch * 10 * 1
    return dice_loss

    
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
dice_loss_every_frame = [dice_loss_1, dice_loss_2, dice_loss_3, dice_loss_4, dice_loss_5,
                        dice_loss_6, dice_loss_7, dice_loss_8, dice_loss_9, dice_loss_10]
#dice_loss_every_frame = [dice_loss_1]