#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %%
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

# %%
cifar_10 = tf.keras.datasets.cifar10
(train_images_former, train_labels_former), (test_images, test_labels) = cifar_10.load_data()

# %%
# Code here!


# In[2]:


train_images_former.shape


# In[3]:


len(train_labels_former)


# In[4]:


type(train_labels_former)


# In[5]:


class_names=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


# In[6]:


# %%
# 画张图看看
plt.figure()
plt.imshow(train_images_former[3])
plt.colorbar()
plt.grid(False)
plt.show()


# In[7]:


# %%
# 像素归一化，归一化后的像素范围为[0, 1]
train_images_former = train_images_former / 255.0

test_images = test_images / 255.0
# %%
# 肉眼可见图变灰了
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images_former[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[int(train_labels_former[i])])
plt.show()


# In[8]:


# %%
# 画张图看看
plt.figure()
plt.imshow(train_images_former[3])
plt.colorbar()
plt.grid(False)
plt.show()


# In[9]:


# 分离train与valid
from sklearn.model_selection import train_test_split
train_images, valid_images, train_labels, valid_labels = train_test_split(train_images_former, train_labels_former, test_size=.2, random_state=1)


# In[10]:


train_images.shape


# In[11]:


len(train_labels)


# In[12]:


# 画图看看分离后的train
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[int(train_labels[i])])
plt.show()


# In[30]:


# 神经网络搭建
cnn_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)), #效果上找出最明显特征
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(2048, activation='relu'),
    tf.keras.layers.Dense(10)
])


# In[31]:


# 回调函数callbacks
from tensorflow.keras import layers, callbacks
early_stopping = callbacks.EarlyStopping(
    min_delta=1e-5, # minimium amount of change to count as an improvement
    patience=20, # how many epochs to wait before stopping
    restore_best_weights=True,
)


# In[32]:


# %%
# 编译模型
# 优化器选择adam（不知道选啥的时候用adam就完事了）
# 损失函数选用SparseCategoricalCrossentropy
cnn_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# In[33]:


# %%
# fit模型，开始训练啦！
history =cnn_model.fit(train_images, train_labels,
                       validation_data=(valid_images, valid_labels), 
                       callbacks=[early_stopping], # callbacks
                       epochs=10, batch_size=128)


# In[34]:


import pandas as pd
history_df = pd.DataFrame(history.history)
print("Minimum validation loss: {}".format(history_df['val_loss'].min()))
history_df.loc[:, ['loss', 'val_loss']].plot()
# %%


# In[35]:


# %%
# 测试模型
test_loss, test_acc = cnn_model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)
# %%
# 这里直接接一个softmax把输出转化成概率
probability_model = tf.keras.Sequential([cnn_model, 
                                         tf.keras.layers.Softmax()])
# %%
predictions = probability_model.predict(test_images)
# %%
predictions[0]
# %%
# argmax把输出最大概率的元素，得到结果
np.argmax(predictions[0])
# %%
test_labels[0]
# %%


# In[36]:


# %%
# 可视化
def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[int(true_label)]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[int(true_label)].set_color('blue')
# %%
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()


# In[20]:


# %%
i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()
# %%
# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()
# %%


# In[ ]:




