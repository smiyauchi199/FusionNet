#import os, sys
#sys.path.append(os.path.dirname(__file__))
import numpy as np
import json
import keras
from models.layers import inputs4d
from models.lvae import lvae_model
from models.processing import load_3ddata, show_result
from models.config import Adam, stoper, reduce_lr
from models.loss import dice_loss_every_frame, dice_loss_sum_3d
from argparse import ArgumentParser
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import h5py
from tensorflow.keras.utils import plot_model
from models.fusion4test import cnn_encoder4plusfusion, cnn_decoder4plusfusion

def main(argv):        
    encoder = cnn_encoder4plusfusion()
    lvae        = lvae_model(add_kl_loss=True)
    decoder     = cnn_decoder4plusfusion()

    f, firstfusion, secondfusion, thirdfusion, fourthfusion  = encoder(inputs4d)
    #z = e
    z, kl_loss = lvae(f)
    output = decoder([z, firstfusion, secondfusion, thirdfusion, fourthfusion])

    model = keras.Model(inputs=inputs4d, outputs=output,name='model')
    model.summary()

    encoder.summary()
    decoder.summary()

    # data
    trdata, tsdata = load_3ddata()
    x = trdata[...,[0,4,8]]
    y = trdata
    print(x.shape)
    print(x.shape)
    print(x.shape)
    batch = 10
    start = int(argv[2])
    end = int(argv[4])
    train_x = np.concatenate((x[0:start], x[end:]), axis=0)
    train_y = np.concatenate((y[0:start], y[end:]), axis=0)
    print(train_x.shape, train_y.shape)

    model.compile(loss=[dice_loss_sum_3d,kl_loss], metrics=dice_loss_every_frame, optimizer=Adam)
    h = model.fit(x=train_x, y=[train_y, train_y], batch_size=10, epochs=350, callbacks=[stoper, reduce_lr])
    h.history.pop('loss')
    h.history.pop('lr')

    # save
    model.save('./result/FusionHybridNetworkInput(with KL)159_{}{}{}.h5'.format(start,'-',end))
    with open('./result/history3d.json', 'w') as f: json.dump(h.history, f)
    def show_result(loss_log, name=None):
        cols=1; rows=1
        fig = plt.figure(7, (cols*16/5,rows*9/5), dpi=300)
        ax = fig.subplots(rows,cols)
        
        pd.DataFrame(loss_log).plot(ax=ax)
        ax.grid(True)  # 方格
        ax.set(ylim=(1e-2, 1e0), xlabel='epoch', ylabel='loss')  # y轴
        ax.legend()
        fig.tight_layout()
        fig.savefig("./result/FusionHybridNetworkInput(with KL)159_{}.png".format(name), dpi=300)
    show_result(h.history , name = str(start) + '-' + str(end))


def read_datas(idx):
    trdata, valdata = load_3ddata(idx)
    x = trdata[...,[0,2,4,6,8]]
    y = trdata
    a = valdata[...,[0,2,4,6,8]]
    b = valdata
    print("trdata shape is ", trdata.shape)
    print("trdata shape is ", trdata.shape)
    print("trdata shape is ", trdata.shape)
    print("valdata shape is ", valdata.shape)
    print("valdata shape is ", valdata.shape)
    print("valdata shape is ", valdata.shape)

    encoder = cnn_encoder4plusfusion()
    lvae        = lvae_model(add_kl_loss=True)
    decoder     = cnn_decoder4plusfusion()

    f, firstfusion, secondfusion, thirdfusion, fourthfusion  = encoder(inputs4d)
    #z = e
    z, kl_loss = lvae(f)
    output = decoder([z, firstfusion, secondfusion, thirdfusion, fourthfusion])

    model = keras.Model(inputs=inputs4d, outputs=output,name='model')
    model.summary()

    encoder.summary()
    decoder.summary()

    def kl_loss(a, b):
        print(b)
        return b
    # data
    batch = 10
    model.compile(loss=[dice_loss_sum_3d,kl_loss], metrics=dice_loss_every_frame, optimizer=Adam)
    h = model.fit(x=x, y=y, validation_data=[a, b], batch_size=10, epochs=500, callbacks=[stoper, reduce_lr])
    h.history.pop('loss')
    h.history.pop('lr')

    # save
    model.save('./Fusion13579_{}{}.h5'.format('datanumber', idx))
    with open('./result/Fusion13579.json', 'w') as f: json.dump(h.history, f)
    def show_result(loss_log, name=None):
        cols=1; rows=1
        fig = plt.figure(7, (cols*16/5,rows*9/5), dpi=300)
        ax = fig.subplots(rows,cols)
        
        pd.DataFrame(loss_log).plot(ax=ax)
        ax.grid(True)  # 方格
        ax.set(ylim=(1e-2, 1e0), xlabel='epoch', ylabel='loss')  # y轴
        ax.legend()
        fig.tight_layout()
        fig.savefig("./Fusion13579_datanumber{}.png".format(name), dpi=300)
    show_result(h.history , name = str(idx))

if __name__ == "__main__":
    import sys
    args = sys.argv
    if len(sys.argv) == 5:
        if sys.argv[1] == "--start" and sys.argv[3] == "--end":
            main(sys.argv)

    if len(sys.argv) == 3:
        if sys.argv[1] == "train":
            idx = args[2]
            read_datas(idx)