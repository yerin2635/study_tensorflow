import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()


# STEP 1 讀取資料
mnist = tf.keras.datasets.mnist
# Tuple of Numpy arrays: (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 將 training 的 input 資料 28*28 的 2維陣列 轉為 1維陣列，再轉成 float32
# 每一個圖片，都變成 784 個 float 的 array
# training 與 testing 資料數量分別是 60000 與 10000 筆
# X_train_2D 是 [60000, 28*28] 的 2維陣列
x_train_2D = x_train.reshape(60000, 28*28).astype('float32')
x_test_2D = x_test.reshape(10000, 28*28).astype('float32')
print('x_train_2D.shape=', x_train_2D.shape)
# x_train_2D.shape=(60000, 784)

# 將圖片的數字 (0~255) 標準化，最簡單的方法就是直接除以 255
# x_train_norm 是標準化後的結果，每一個數字介於 0~1 之間
x_train_norm = x_train_2D/255
x_test_norm = x_test_2D/255

# 將 training 的 label 進行 one-hot encoding，例如數字 7 經過 One-hot encoding 轉換後是 array([0., 0., 0., 0., 0., 0., 0., 1., 0., 0.], dtype=float32)，即第7個值為 1

y_train_one_hot_tf=tf.one_hot(y_train,10)
y_test_one_hot_tf=tf.one_hot(y_test,10)

y_train_one_hot = None
y_test_one_hot = None
with tf.compat.v1.Session() as sess:
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)
    y_train_one_hot = sess.run(y_train_one_hot_tf)
    y_test_one_hot = sess.run(y_test_one_hot_tf)

# 將 x_train, y_train 分成 train 與 validation 兩個部分
x_train_norm_data = x_train_norm[0:50000]
x_train_norm_validation = x_train_norm[50000:60000]

y_train_one_hot_data = y_train_one_hot[0:50000]
y_train_one_hot_validation = y_train_one_hot[50000:60000]


### 建立模型

# 先建立一些共用的函數
def weight(shape):
    return tf.Variable(tf.random.truncated_normal(shape, stddev=0.1),
                       name ='W')
# bias 張量，先以 constant 建立常數，然後用 Variable 建立張量變數
def bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape)
                       , name = 'b')
# 卷積運算 功能相當於濾鏡
#  x 是輸入的影像，必須是 4 維的張量
#  W 是 filter weight 濾鏡的權重，後續以隨機方式產生 filter weight
#  strides 是 濾鏡的跨步 step，設定為 [1,1,1,1]，格式是 [1, stride, stride, 1]，濾鏡每次移動時，從左到右，上到下，各移動 1 步
#  padding 是 'SAME'，此模式會在邊界以外 補0 再做運算，讓輸入與輸出影像為相同大小
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1],
                        padding='SAME')

# 建立池化層，進行影像的縮減取樣
#  x 是輸入的影像，必須是 4 維的張量
#  ksize 是縮減取樣窗口的大小，設定為 [1,2,2,1]，格式為 [1, height, width, 1]，也就是高度 2 寬度 2 的窗口
#  stides 是縮減取樣窗口的跨步 step，設定為 [1,2,2,1]，格式為 [1, stride, stride, 1]，也就是縮減取樣窗口，由左到右，由上到下，各2步
#  原本 28x28 的影像，經過 max-pool 後，會縮小為 14x14
def max_pool_2x2(x):
    return tf.nn.max_pool2d(x, ksize=[1,2,2,1],
                          strides=[1,2,2,1],
                          padding='SAME')


# 輸入層
with tf.name_scope('Input_Layer'):
    # placeholder 會傳入影像
    x = tf.compat.v1.placeholder("float",shape=[None, 784],name="x")
    # x 原本為 1 維張量，要 reshape 為 4 維張量
    # 第 1 維 -1，因為後續訓練要透過 placeholder 輸入的資料筆數不固定
    # 第 2, 3 維，是 28, 28，因為影像為 28x28
    # 第 4 維是 1，因為是單色的影像，就設定為 1，如果是彩色，要設定為 3 (RGB)
    x_image = tf.reshape(x, [-1, 28, 28, 1])

# CNN Layer 1
# 用來提取特徵，卷積運算後，會產生 16 個影像，大小仍為 28x28
with tf.name_scope('C1_Conv'):
    # filter weight 大小為 5x5
    # 因為是單色，第 3 維設定為 1
    # 要產生 16 個影像，所以第 4 維設定為 16
    W1 = weight([5,5,1,16])

    # 因為產生 16 個影像，所以輸入餐次 shape = 16
    b1 = bias([16])

    # 卷積運算
    Conv1=conv2d(x_image, W1)+ b1
    # ReLU 激活函數
    C1_Conv = tf.nn.relu(Conv1 )

# 池化層用來 downsampling，將影像由 28x28 縮小為 14x14，影像數量仍為 16
with tf.name_scope('C1_Pool'):
    C1_Pool = max_pool_2x2(C1_Conv)

# CNN Layer 2
# 第二次卷積運算，將 16 個影像轉換為 36 個影像，卷積運算不改變影像大小，仍為 14x14
with tf.name_scope('C2_Conv'):
    # filter weight 大小為 5x5
    # 第 3 維是 16，因為卷積層1 的影像數量為 16
    # 第 4 維設定為 36，因為將 16 個影像轉換為 36個
    W2 = weight([5,5,16,36])
    # 因為產生 36 個影像，所以輸入餐次 shape = 36
    b2 = bias([36])
    Conv2=conv2d(C1_Pool, W2)+ b2
    # relu 會將負數的點轉換為 0
    C2_Conv = tf.nn.relu(Conv2)

# 池化層2用來 downsampling，將影像由 14x14 縮小為 7x7，影像數量仍為 36
with tf.name_scope('C2_Pool'):
    C2_Pool = max_pool_2x2(C2_Conv)

# Fully Connected Layer
# 平坦層，將 36個 7x7 影像，轉換為 1 維向量，長度為 36x7x7= 1764，也就是 1764 個 float，作為輸入資料
with tf.name_scope('D_Flat'):
    D_Flat = tf.reshape(C2_Pool, [-1, 1764])

with tf.name_scope('D_Hidden_Layer'):
    W3= weight([1764, 128])
    b3= bias([128])
    D_Hidden = tf.nn.relu(
                  tf.matmul(D_Flat, W3)+b3)

    ## Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
    # D_Hidden_Dropout= tf.nn.dropout(D_Hidden, keep_prob=0.8)
    D_Hidden_Dropout= tf.nn.dropout(D_Hidden, rate = 0.2)

# 輸出層, 10 個神經元
#  y_predict = softmax(D_Hidden_Dropout * W4 + b4)
with tf.name_scope('Output_Layer'):
    # 因為上一層 D_Hidden 是 128 個神經元，所以第1維是 128
    W4 = weight([128,10])
    b4 = bias([10])
    y_predict= tf.nn.softmax(
                 tf.matmul(D_Hidden_Dropout, W4)+b4)


### 設定訓練模型最佳化步驟
# 使用反向傳播演算法，訓練多層感知模型
with tf.name_scope("optimizer"):

    y_label = tf.compat.v1.placeholder("float", shape=[None, 10],
                              name="y_label")

    loss_function = tf.reduce_mean(
                      tf.nn.softmax_cross_entropy_with_logits
                         (logits=y_predict ,
                          labels=y_label))

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001) \
                    .minimize(loss_function)


### 設定評估模型
with tf.name_scope("evaluate_model"):
    correct_prediction = tf.equal(tf.argmax(y_predict, 1),
                                  tf.argmax(y_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


### 訓練模型

trainEpochs = 5
batchSize = 100
totalBatchs = int(len(x_train_norm_data)/batchSize)
epoch_list=[];accuracy_list=[];loss_list=[];
from time import time

with tf.compat.v1.Session() as sess:
    startTime=time()

    sess.run(tf.compat.v1.global_variables_initializer())

    for epoch in range(trainEpochs):
        for i in range(totalBatchs):
            # batch_x, batch_y = mnist.train.next_batch(batchSize)
            batch_x = x_train_norm_data[i*batchSize:(i+1)*batchSize]
            batch_y = y_train_one_hot_data[i*batchSize:(i+1)*batchSize]

            sess.run(optimizer,feed_dict={x: batch_x,
                                          y_label: batch_y})

        loss,acc = sess.run([loss_function,accuracy],
                            feed_dict={x: x_train_norm_validation,
                                       y_label: y_train_one_hot_validation})

        epoch_list.append(epoch)
        loss_list.append(loss)
        accuracy_list.append(acc)

        print("Train Epoch:", '%02d' % (epoch+1), "Loss=","{:.9f}".format(loss)," Accuracy=",acc)

    duration =time()-startTime
    print("Train Finished takes:",duration)

    ## 評估模型準確率
    print("Accuracy:",
      sess.run(accuracy,feed_dict={x: x_test_norm,
                                   y_label:y_test_one_hot}))
    # 前 5000 筆
    print("Accuracy:",
      sess.run(accuracy,feed_dict={x: x_test_norm[:5000],
                                   y_label: y_test_one_hot[:5000]}))
    # 後 5000 筆
    print("Accuracy:",
      sess.run(accuracy,feed_dict={x: x_test_norm[5000:],
                                   y_label: y_test_one_hot[5000:]}))

    ## 預測機率
    y_predict=sess.run(y_predict,
                   feed_dict={x: x_test_norm[:5000]})

    ## 預測結果
    prediction_result=sess.run(tf.argmax(y_predict,1),
                           feed_dict={x: x_test_norm ,
                                      y_label: y_test_one_hot})

    ## 儲存模型
    saver = tf.compat.v1.train.Saver()
    save_path = saver.save(sess, "handwritten_numeral_recognition/saveModel/CNN_model1")
    print("Model saved in file: %s" % save_path)
    merged = tf.compat.v1.summary.merge_all()
    # 可將 計算圖，透過 TensorBoard 視覺化
    train_writer = tf.compat.v1.summary.FileWriter('handwritten_numeral_recognition/log/CNN',sess.graph)


# matplotlib 列印 loss, accuracy 折線圖
import matplotlib.pyplot as plt

fig = plt.gcf()
# fig.set_size_inches(4,2)
plt.plot(epoch_list, loss_list, label = 'loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss'], loc='upper left')
plt.savefig('handwritten_numeral_recognition/loss.png')


fig = plt.gcf()
# fig.set_size_inches(4,2)
plt.plot(epoch_list, accuracy_list,label="accuracy" )

plt.ylim(0.8,1)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['accuracy'], loc='upper right')
plt.savefig('handwritten_numeral_recognition/accuracy.png')

############
# 查看多筆資料，以及 label
import matplotlib.pyplot as plt
def plot_images_labels_prediction(images,labels,prediction,idx,filename, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num>25: num=25
    for i in range(0, num):
        ax=plt.subplot(5,5, 1+i)

        # 將 images 的 784 個數字轉換為 28x28
        ax.imshow(np.reshape(images[idx],(28, 28)), cmap='binary')

        # 轉換 one_hot label 為數字
        title= "label=" +str(np.argmax(labels[idx]))
        if len(prediction)>0:
            title+=",predict="+str(prediction[idx])

        ax.set_title(title,fontsize=10)
        ax.set_xticks([]);ax.set_yticks([])
        idx+=1
    plt.savefig(filename)


plot_images_labels_prediction(x_test_norm,
                              y_test_one_hot,
                              prediction_result,0, "handwritten_numeral_recognition/result.png", num=10)

# 找出預測錯誤
for i in range(400):
    if prediction_result[i]!=np.argmax(y_test_one_hot[i]):
        print("i="+str(i)+
              "   label=",np.argmax(y_test_one_hot[i]),
              "predict=",prediction_result[i])