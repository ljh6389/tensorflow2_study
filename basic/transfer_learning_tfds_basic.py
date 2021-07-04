'''
 tensorflow 공식 tutorial 참고
 https://www.tensorflow.org/tutorials/images/transfer_learning?hl=ko
'''
import tensorflow as tf
import tensorflow_datasets as tfds

tfds.disable_progress_bar()

(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

IMG_SIZE = 160

def format_example(image, label):
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1.
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

train_data = raw_train.map(format_example)
val_data = raw_validation.map(format_example)
test_data = raw_test.map(format_example)

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train_data.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
val_batches = val_data.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_batches = test_data.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

for image_batch, label_batch in train_batches.take(1):
   pass

print('batch_shape...', image_batch.shape)

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

load_w_top = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=True,
                                               weights='imagenet')

load_wo_top = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                include_top=False,
                                                weights='imagenet')

output = load_w_top.layers[-3].output
feat_head_w_top = tf.keras.Model(inputs=load_w_top.input, outputs=output)
feat_head_w_top.summary()

output = load_wo_top.layers[-1].output
feat_head_wo_top = tf.keras.Model(inputs=load_wo_top.input, outputs=output)
feat_head_wo_top.summary()

feat_head_w_top.trainable = False
feat_head_wo_top.trainable = False




