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

'''
 전체 layer를 불러오는 경우의 예시
 - custom pretrained weight파일을 쓰는 경우, top layer를 제외하고 불러올 경우 네트워크 구조가 안맞는 에러가 발생
 - 따라서, 아래와 같이 전체 네트워크를 불러온 뒤 classification head만 따로 떼어내는 작업을 진 
'''
load_w_top = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=True,
                                               weights='imagenet')

output = load_w_top.layers[-3].output
feat_head_w_top = tf.keras.Model(inputs=load_w_top.input, outputs=output)
feat_head_w_top.trainable = False

cls_w_top = tf.keras.Sequential([
    feat_head_w_top,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1)
])
cls_w_top.summary()

cls_w_top.compile(
    optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)

print('================include_top=True model training and eval======================')
cls_w_top.summary()

cls_w_top.fit(
    train_batches,
    epochs=1,
    validation_data=val_batches
)
cls_w_top.evaluate(test_batches, verbose=2)
print('========================================================================\n')

'''
keras에서 제공하는 pretrained weight를 사용할 경우 include_top option을 False로 줌을 통해 간단하게 backbone weight만 로드 가능
'''
load_wo_top = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                include_top=False,
                                                weights='imagenet')

output = load_wo_top.layers[-1].output
feat_head_wo_top = tf.keras.Model(inputs=load_wo_top.input, outputs=output)
feat_head_wo_top.trainable = False

cls_wo_top = tf.keras.Sequential([
    feat_head_wo_top,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

cls_wo_top.compile(
    optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=['accuracy']
)

print('================include_top=False model training and eval======================')
cls_wo_top.summary()

cls_wo_top.fit(
    train_batches,
    epochs=1,
    validation_data=val_batches
)

cls_wo_top.evaluate(test_batches, verbose=2)
print('========================================================================\n')
