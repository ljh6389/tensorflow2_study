'''
 tensorflow 공식 tutorial 참고
 https://www.tensorflow.org/tutorials/quickstart/beginner?hl=ko
'''

import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

'''
255로 나누는 방법으로 input normalization 진행
'''
x_train = x_train / 255.
x_test = x_test / 255.


'''
tf.keras.models.Sequential을 이용한 모델 선언
참고: https://www.tensorflow.org/tutorials/quickstart/beginner?hl=ko
'''
sequential_model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

sequential_model.compile(optimizer='adam',
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])
print('================sequential model training and eval======================')
sequential_model.fit(x_train, y_train, epochs=5)
sequential_model.evaluate(x_test, y_test, verbose=2)
print('========================================================================\n')


'''
Functional API 활용 (tf.keras.Model)
참고 : https://www.tensorflow.org/api_docs/python/tf/keras/Model
'''
inputs = tf.keras.Input(shape=(28, 28))
x = tf.keras.layers.Flatten(input_shape=(28, 28))(inputs)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
functional_model = tf.keras.Model(inputs=inputs, outputs=outputs)
functional_model.compile(optimizer='adam',
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])

print('================sequential model training and eval======================')
functional_model.fit(x_train, y_train, epochs=5)
functional_model.evaluate(x_test, y_test, verbose=2)
print('========================================================================\n')


'''
subclassing 활용 (tf.keras.Model 상속한 class 생성)
참고 : https://www.tensorflow.org/api_docs/python/tf/keras/Model
'''
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.out_layer = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False, mask=None):
        x = self.flatten(inputs)
        x = self.dense(x)

        if training:
            x = self.dropout(x)

        return self.out_layer(x)

sub_model = MyModel()

sub_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

print('================subclassing model training and eval======================')
sub_model.fit(x_train, y_train, epochs=5)
sub_model.evaluate(x_test, y_test, verbose=2)
print('========================================================================\n')
