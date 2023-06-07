#import os, sys
#sys.path.append(os.path.dirname(__file__))
import numpy as np
import json
import keras
from models import inputs4d, lvae_model, load_3ddata, \
                   Adam, stoper, reduce_lr, show_result
from models.loss import dice_loss_every_frame, dice_loss_sum_3d
from argparse import ArgumentParser
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import h5py
from tensorflow.keras.utils import plot_model
from models.fusion4test import cnn_encoder4plusfusion, cnn_decoder4plusfusion
# model 初期化


def main(argv):        
    encoder = fusion3Frameencoder()
    lvae        = lvae_model(add_kl_loss=True)
    decoder     = fusion3Framedecoder()

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

    # train_x = x[10:]
    # train_y = y[10:]

    # val_x = x[:10]
    # val_y = y[:10]

    # val_x = x[start:end]
    # val_y = y[start:end]

    # train_x = list(np.split(train_x, train_x.shape[0]//10))
    # train_y = list(np.split(train_y, train_y.shape[0]//10))

    # def gen():
    #     for i, j in zip(train_x, train_y):
    #         yield (i, j)

    # dataset = tf.data.Dataset.from_generator(gen, output_signature=(
    #          tf.TensorSpec(shape=(None, 80, 80, 80, 5), dtype=tf.float32),
    #          tf.TensorSpec(shape=(None, 80, 80, 80, 10), dtype=tf.float32)))
    # train

    model.compile(loss=[dice_loss_sum_3d,kl_loss], metrics=dice_loss_every_frame, optimizer=Adam)
    h = model.fit(x=train_x, y=[train_y, train_y], batch_size=10, epochs=350, callbacks=[stoper, reduce_lr])
    #h = model.fit(dataset, batch_size=10, epochs=25, validation_data=(val_x, val_y), callbacks=[stoper, reduce_lr])
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
        #plt.gca().set_xlim(0, 140)
        ax.legend()
        fig.tight_layout()
        fig.savefig("./result/FusionHybridNetworkInput(with KL)159_{}.png".format(name), dpi=300)
        #plt.show()
    show_result(h.history , name = str(start) + '-' + str(end))
# def kl_loss(a, b):
#     print(b)
#     return b

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
    #model.compile(loss=dice_loss_sum_3d, metrics=dice_loss_every_frame, optimizer=Adam)
    model.compile(loss=[dice_loss_sum_3d,kl_loss], metrics=dice_loss_every_frame, optimizer=Adam)
    h = model.fit(x=x, y=y, validation_data=[a, b], batch_size=10, epochs=500, callbacks=[stoper, reduce_lr])
    #h = model.fit(dataset, batch_size=10, epochs=25, validation_data=(val_x, val_y), callbacks=[stoper, reduce_lr])
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
        #plt.gca().set_xlim(0, 140)
        ax.legend()
        fig.tight_layout()
        fig.savefig("./Fusion13579_datanumber{}.png".format(name), dpi=300)
        #plt.show()
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