## NN Model Template

Template Kernel：<https://www.kaggle.com/fantasticgold/nn-model?scriptVersionId=15254431>



### Baseline

**Features** - features_12

**Indexs** - random_index

**Parameters** - 

+ batch_size = 32
+ learning_rate = 5e-4
+ steps_per_epoch = 1000
+ n_epochs = 30
+ validation_steps = 200

**Datasets** - 

+ train = [0, ]
+ validation = [0, ]

**Model** - GRU

**Score** - 1.55



### More

| Date      | Features   | Indexs              | Parameters                           | Datasets   |   Model   | Public Score | Private Score |
| --------- | ---------- | ------------------- | ------------------------------------ | ---------- | --------- | ------------ | ------------- |
| 2019.6.5  |            | random_index        |                                      |            |   LSTM    |     1.52     |               |
| 2019.6.6  |            | random_index_serial |                                      |            |   LSTM    |     1.48     |               |
| 2019.6.7  |            | random_index_serial | n_epochs = 100                       |            |   LSTM    |     1.59     |               |
| 2019.6.9  |            | random_index_serial | ........                             | ....       |   GRU     |     1.53     |               |
| 2019.6.9  |            | random_index_serial | lr=0.01*(0.2^n)                      | ....       | GRU+LSTM  |    1.50224   |    2.56412    |
| 2019.6.10 |            | random_index_serial | batch_size = 64 learning_rate = 5e-3 |            |   LSTM    |     1.65     |               |
| 2019.6.1 | feature_15 | random_index_serial |                                      |            |   LSTM    |    1.50047   |    2.74354    |
| 2019.6.12 |            | random_index_serial |                                      | train=[1,] |   LSTM    |     1.54     |               |
| 2019.6.13 | feature_15 | random_index_serial ||||||
| 2019.6.14 | feature_16 | random_index_serial |                                      | train=[1,] |   LSTM    |     1.51     |               |
| 2019.6.15 |            |                     | Use 1st team submit                  |            | Use 1st team's submit |     60       |     .....     |

#### Further details

##### 2019.6.9

Add callbacks

```python
cb = [ModelCheckpoint("model.hdf5", save_best_only=True, period=3),
      # ==> will stop when val_loss doesn't improve in 5 epochs
      EarlyStopping(monitor='val_loss', patience=5),                
      # ==> will make lr = 0.2 * lr when loss doesn't improve in 5 epochs
      ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=10e-8)]
```

model structure change, add GRU layers.

```python
model.add(CuDNNGRU(48, input_shape=(None, n_features), return_sequences=True))
model.add(CuDNNGRU(48, input_shape=(None, 48)))
model.add(Dense(10, activation="relu"))
model.add(Dense(1))
model.compile(optimizer=adam(lr=learning_rate), loss="mae")
```

draw plot about loss

```python
history = model.fit_generator(...)

# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
```

add a new feature (not sure whether it works)



## 个人总结

- 黄嘉杰
  - 在调参时，因为是在kaggle上跑的，每天只能交两次，效率不是很高
  - 另外感觉是因为调用了Keras的API，运行的结果有一定的随机性，我曾试过用同一份代码跑四次，每次的结果都不一样，最低的是1.51，最高的是1.65，大大增加了调参的难度
  - 最开始先简单地将n_epochs调大，试了几次发现效果反而更差了，因为不能确定是不是随机性导致的，后面没有继续调下去，使用了一个比较小的n_epochs
  - 对于训练集要不要包含验证集的问题，我更倾向于不包含，从结果上对比是差不多的
  - 对于RNN、GRU、LSTM的对比，从结果来看LSTM相对会更好一点，感觉在处理时间序列的问题上LSTM会更有优势
  - 对于特征的提取，还是用的基本的四种均值、标准差、最大值、最小值，曾经考虑了一些拓展的特征，比如拟合曲线的斜率、大于某个阈值的信号量的百分比等，但是没有实现
  - 还考虑过用CNN做，一种思路是将这些信号随机分段，然后与验证集做匹配；另外一种思路是先用CNN提取特征，再用RNN/GRU/LSTM来训练；因为看其他Kernal里的分数还不如我们现在做的，所以也没有继续做

