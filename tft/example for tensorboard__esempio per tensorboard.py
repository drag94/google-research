import tensorflow as tf
import datetime

if __name__ == '__main__':

    class CustomCallback(tf.keras.callbacks.Callback):
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

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    log_dir = "/home/davide/PycharmProjects/tensorflow2-example/tensorboard-logs/" + \
              datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
    file_writer.set_as_default()

    # TODO: Exponential decay
    def lr_schedule(epoch):
        """
        Returns a custom learning rate that decreases as epochs progress.
        """
        learning_rate = 0.2
        if epoch > 10:
            learning_rate = 0.02
        if epoch > 20:
            learning_rate = 0.01
        if epoch > 50:
            learning_rate = 0.005

        tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
        return learning_rate


    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    model.compile(optimizer='SGD',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=2,
              validation_data=(x_test, y_test),
              callbacks=[tensorboard_callback, lr_callback, CustomCallback()])

    # tf.summary.scalar('learning rate/val_loss', data=model.history['val_loss'], step=epoch)
    # tf.summary.scalar('learning rate/val_accuracy', data=model.history['val_accuracy'], step=epoch)

    model.evaluate(x_test, y_test, verbose=2)
