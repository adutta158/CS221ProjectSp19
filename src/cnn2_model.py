from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
import util
import os
from base_model import BaseModel

class Cnn2Model(BaseModel):
    """Implement CNN2
    Example usage:
    > model = Cnn2Model()
    > model.train(x_train, y_train, batch_size, epochs)
    > model.predict(x_eval)
    """
    def __init__(self, step_size=0.2, max_iter=1e5, threshold=1e-5, verbose=False, shape = (28,28,1), classes = 22):
        BaseModel.__init__(self, step_size, max_iter, threshold, verbose)
        # Create model architecture
        self.clf = Sequential()
        self.clf.add(Conv2D(32, (5, 5), input_shape=shape, activation='relu'))
        self.clf.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
        self.clf.add(Conv2D(64, (5, 5), activation='relu'))
        self.clf.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
        self.clf.add(Flatten())
        self.clf.add(Dense(512, activation='relu'))
        self.clf.add(Dropout(0.5))
        self.clf.add(Dense(128, activation='relu'))
        self.clf.add(Dropout(0.5))
        self.clf.add(Dense(classes, activation='softmax'))

        # Compile the model
        self.clf.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.clf.summary()
        self.C = classes

    def train(self, x, y, filename = "mlp", batch = 32, epoch = 1):
        """fit classifier
        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        if os.path.isfile(filename + '.h5'):
            self.clf = load_model(filename + '.h5')
        else:
            self.clf.fit(x.reshape(x.shape[0], 28, 28, 1), util.one_hot(y, self.C), batch_size=batch, epochs=epoch)
            self.clf.save(filename + '.h5')
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.
                m - number of samples in the set x
                n - number of features
                C - number of classes
                Args:
                    x: Inputs of shape (m, n).
                    Returns:
                    Outputs of shape (m, 1)
        """
        # *** START CODE HERE ***
        return self.clf.predict_classes(x.reshape(x.shape[0], 28, 28, 1))
        # *** END CODE HERE ***

