import tensorflow as tf
from tensorflow import keras

if __name__='__main__':

    class CustomCallback(keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            keys = list(logs.keys())
            print("Starting training; got log keys: {}".format(keys))

        def on_train_end(self, logs=None):
            keys = list(logs.keys())
            print("Stop training; got log keys: {}".format(keys))

        def on_epoch_begin(self, epoch, logs=None):
            keys = list(logs.keys())
            print("Start epoch {} of training; got log keys: {}".format(epoch, keys))

        def on_epoch_end(self, epoch, logs=None):
            print("End epoch {} of training; got log keys: {}".format(epoch, logs))

            tf.summary.scalar('validation/val_loss', data=logs['val_loss'], step=epoch)
            tf.summary.scalar('validation/val_accuracy', data=logs['val_accuracy'], step=epoch)
            tf.summary.flush()

        def on_test_begin(self, logs=None):
            keys = list(logs.keys())
            print("Start testing; got log keys: {}".format(keys))

        def on_test_end(self, logs=None):
            keys = list(logs.keys())
            print("Stop testing; got log keys: {}".format(keys))

        def on_predict_begin(self, logs=None):
            keys = list(logs.keys())
            print("Start predicting; got log keys: {}".format(keys))

        def on_predict_end(self, logs=None):
            keys = list(logs.keys())
            print("Stop predicting; got log keys: {}".format(keys))

        def on_train_batch_begin(self, batch, logs=None):
            keys = list(logs.keys())
            print("...Training: start of batch {}; got log keys: {}".format(batch, keys))

        def on_train_batch_end(self, batch, logs=None):
            keys = list(logs.keys())
            print("...Training: end of batch {}; got log keys: {}".format(batch, keys))

        def on_test_batch_begin(self, batch, logs=None):
            keys = list(logs.keys())
            print("...Evaluating: start of batch {}; got log keys: {}".format(batch, keys))

        def on_test_batch_end(self, batch, logs=None):
            keys = list(logs.keys())
            print("...Evaluating: end of batch {}; got log keys: {}".format(batch, keys))

        def on_predict_batch_begin(self, batch, logs=None):
            keys = list(logs.keys())
            print("...Predicting: start of batch {}; got log keys: {}".format(batch, keys))

        def on_predict_batch_end(self, batch, logs=None):
            keys = list(logs.keys())
            print("...Predicting: end of batch {}; got log keys: {}".format(batch, keys))