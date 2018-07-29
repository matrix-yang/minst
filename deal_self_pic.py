# coding: utf-8
# 将原始图片转换成需要的大小，并将其保存
# ========================================================================================
import os
import tensorflow as tf
from PIL import Image
import os.path
import glob
import matplotlib.pyplot as plt

root_dir = "C:\\Users\\Matrix-yang\\Desktop\\birds\\train"


def create_tf_record():
    writer = tf.python_io.TFRecordWriter("bird_train.tfrecords")
    # 获取根目录下所有子目录（第一个获取到的目录为根目录）
    sub_dirs = [x[0] for x in os.walk(root_dir)]

    is_root_dir = True
    for sub_dir in sub_dirs:
        # 跳过根目录
        if is_root_dir:
            is_root_dir = False
            continue
        # os.path.basename(sub_dir)  获取最后一级目录作为label
        label = os.path.basename(sub_dir)
        file_path_list = []
        extensions = ['jpg', 'jpeg']
        for e in extensions:
            file_golb = os.path.join(sub_dir, "*." + e)
            file_path_list.extend(glob.glob(file_golb))
        if file_path_list:
            for file_path in file_path_list:
                img = Image.open(file_path)
                img = img.resize((256, 256))  # 设置需要转换的图片大小
                #不是rgb的图像抛弃
                if (img.mode != "RGB"):
                    print(img.mode, img)
                    plt.imshow(img)
                    plt.show()
                    continue

                img_raw = img.tobytes()
                # print(label,img_raw)
                example = tf.train.Example(
                    features=tf.train.Features(feature={
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label)])),
                        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                    }))
                writer.write(example.SerializeToString())
    writer.close()


def read_and_decode(filename):
    # 创建文件队列,不限读取的数量
    filename_queue = tf.train.string_input_producer([filename])
    # create a reader from file queue
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    # 解析符号化的样本
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string)
        })
    label = features['label']
    img = features['img_raw']
    img = tf.decode_raw(img, tf.uint8)
    # try:
    #     img = tf.reshape(img, [256, 256, 3])
    # except:
    #     img
    # img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(label, tf.int32)
    return img, label


if __name__ == '__main__':
    create_tf_record()

if __name__ == '__main1__':
    batch = read_and_decode('bird_train.tfrecords')
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session() as sess:  # 开始一个会话
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(1000):
            img, label = sess.run(batch)  # 在会话中取出image和label
            print(i, img.shape)
            # plt.imshow(img)
            # plt.show()

        coord.request_stop()
        coord.join(threads)
        sess.close()

# 原始图片的存储位置
orig_picture = 'E:/train_test/train_data/generate_sample/'

# 生成图片的存储位置
gen_picture = 'E:/Re_train/image_data/inputdata/'

# 需要的识别类型
classes = {'husky', 'jiwawa', 'poodle', 'qiutian'}

# 样本总数
num_samples = 120


# =======================================================================================
def read_and_decode(filename):
    # 创建文件队列,不限读取的数量
    filename_queue = tf.train.string_input_producer([filename])
    # create a reader from file queue
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    # 解析符号化的样本
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string)
        })
    label = features['label']
    img = features['img_raw']
    img = tf.decode_raw(img, tf.uint8)
    img = tf.reshape(img, [64, 64, 3])
    # img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(label, tf.int32)
    return img, label


# =======================================================================================
'''
if __name__ == '__main__':
    create_record()
    batch = read_and_decode('dog_train.tfrecords')
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session() as sess: #开始一个会话
        sess.run(init_op)
        coord=tf.train.Coordinator()
        threads= tf.train.start_queue_runners(coord=coord)

        for i in range(num_samples):
            example, lab = sess.run(batch)#在会话中取出image和label
            img=Image.fromarray(example, 'RGB')#这里Image是之前提到的
            img.save(gen_picture+'/'+str(i)+'samples'+str(lab)+'.jpg')#存下图片;注意cwd后边加上‘/’
            print(example, lab)
        coord.request_stop()
        coord.join(threads)
        sess.close()
'''

# ========================================================================================
