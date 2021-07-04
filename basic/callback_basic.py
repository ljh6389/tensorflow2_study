import tensorflow as tf
import numpy as np

#ealry stopping callback
earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

#LR_scheduler
def scheduler(epoch):
    if epoch < 10:
        return 0.001
    else:
        return 0.001 * tf.math.exp(0.1 * (10-epoch))

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

#ModelCheckpoint
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='./model/checkpoint.{epoch:02d}.hdf5',
    save_weights_only=True,
    save_freq='epoch'
)

#LambdaCallback
log = tf.keras.callbacks.LambdaCallback(on_epoch_begin=lambda epoch, logs: print('wwww epoch {}'.format(epoch+1)))

'''
model.fit(....,callbacks=[
    earlystop, lr_scheduler, checkpoint, log
])
'''

class CustomLRscheduler(tf.keras.callbacks.Callback):
    def __init__(self, patience=0):
        super(CustomLRscheduler, self).__init__()
        self.patience = patience
        self.best_weights = None

    def on_train_begin(self, logs=None):
        #The number of epoch it has waited when loss is no longer min
        self.wait = 0
        # The epoch the training stops at
        self.stopped_epoch = 0
        # init best loss
        self.best_loss = -1.*float('inf')
        self.best_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            current_acc = logs.get('val_acc')
            if current_acc > self.best_acc:#np.greater(current_acc, self.best_acc)
                self.best_acc = current_acc
                self.best_weights = self.model.get_weights()

        current_loss = logs.get('loss')
        if np.less(current_loss, self.best_loss):
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('early stop at {:5d}'.format(self.stopped_epoch))

callback1 = tf.keras.callbacks.ModelCheckpoint(
    filepath='./model/checkpoint.{epoch:02d}.hdf5',
    monitor='val_acc',
    save_best_only=True,
    save_freq=10
)
callback2 = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)

custom_callback = [callback1, callback2]