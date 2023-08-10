from tensorflow import keras

stoper = keras.callbacks.EarlyStopping(
	monitor='loss',
	patience=15, 
	verbose=0,
	mode='auto'
	)


lr_conf={'lr' : [1e-3,0.5e-3,1e-4,0.5e-4], 'boundary':[40,70,100]}
def lr_schedule(epoch):
    if epoch < lr_conf['boundary'][0]:lr = lr_conf['lr'][0]
    elif epoch < lr_conf['boundary'][1]:lr = lr_conf['lr'][1]
    elif epoch < lr_conf['boundary'][2]:lr = lr_conf['lr'][2]
    else: lr = lr_conf['lr'][3]
    return lr

lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='loss',
    factor=0.5,
    patience=15,
    verbose=0,
    mode='auto',
    epsilon=0.1,
    cooldown=0,
    min_lr=1e-5
)

Adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=False)

boundaries=[40, 80, 100, 120]
values=[1e-3, 0.5e-3, 1e-4, 0.5e-4, 1e-5]
piece_wise_constant_decay = keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=boundaries, values=values, name=None)
Adam2 = keras.optimizers.Adam(piece_wise_constant_decay)