from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import h5py
import os

def show_result_picture(model, ts_Data_x, ts_Data_y, showfigure=True):
    y   = model.predict(ts_Data_x)[0]
    color_map   = np.stack([ts_Data_y[0], y, y*ts_Data_y[0]],axis=-1)
    plt.figure()
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.imshow(color_map[:,:,i,:])
        plt.axis('off')
        plt.title(i)#kl_score: 1 to 10
    if showfigure:plt.show()

def show_figure(loss_log,index=None):
    pd.DataFrame(loss_log).plot(figsize=(5, 5))
    plt.grid(True)
    plt.gca().set_ylim(1e-2, 1e0)  # y轴
    plt.gca().set_xlim(0, 140)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    if index:plt.savefig("./result/dice(%d).png"%index)
    else: plt.show()
    #plt.clf()
    #plt.show()

# def load_3ddata():
#     Path_to_trainingDatas = "./datas/segmentation_train.h5"
#     Path_to_testingDatas = "./datas/segmentation_test.h5"
#     tr_File = h5py.File(Path_to_trainingDatas, 'r')
#     ts_File = h5py.File(Path_to_testingDatas, 'r')

#     tr_Data = tr_File['train'][:]
#     ts_Data = ts_File['test'][:]

#     # to close the file
#     tr_File.close()
#     ts_File.close()
#     return tr_Data, ts_Data

def load_3ddata(idx: int):
    # ----Path_to_trainingDatas = "./datas/segmentation_train.h5"
    # tr_File = np.load('./datas/x_train_after_registration_0-180.npz')
    # Path_to_testingDatas = "./datas/segmentation_test.h5"
    # ----tr_File = h5py.File(Path_to_trainingDatas, 'r')
    # ts_File = h5py.File(Path_to_testingDatas, 'r')

#-------------------------------
    fname_test = f"test{idx}.npy"
    fanme_val = f"val{idx}.npy"
    fname_train = f"trainaug{idx}.npz"

    dataset_path = os.path.join(os.path.abspath("./new_dataset_(Forallmodels)"), idx)

    x_train = np.load(os.path.join(dataset_path, fname_train))
    x_train = x_train['arr_0']

    x_val = np.load(os.path.join(dataset_path, fanme_val))
    # x_test = np.load(os.path.join(dataset_path, fname_test))
#-------------------------------
    #tr_Data = tr_File
    # tr_Data = tr_File['arr_0'][:]
    # print(tr_Data.shape)
    # ts_Data = ts_File['test'][:]

    print(x_train.shape)
    print(x_val.shape)

    # to close the file
    # x_train.close()
    # x_val.close()
    return x_train, x_val

def show_result(loss_log):
    cols=1; rows=1
    fig = plt.figure(7, (cols*16/5,rows*9/5), dpi=300)
    ax = fig.subplots(rows,cols)
    
    pd.DataFrame(loss_log).plot(ax=ax)
    ax.grid(True)  # 方格
    ax.set(ylim=(1e-2, 1e0), xlabel='epoch', ylabel='loss')  # y轴
    #plt.gca().set_xlim(0, 140)
    ax.legend()
    fig.tight_layout()
    fig.savefig("./result/dice.png", dpi=300)
    plt.show()