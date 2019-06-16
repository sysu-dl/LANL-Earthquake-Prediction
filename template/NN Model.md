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

| Date      | Features   | Indexs              | Parameters                           | Datasets   | Model | Score |
| --------- | ---------- | ------------------- | ------------------------------------ | ---------- | ----- | ----- |
| 2019.6.5  |            | random_index        |                                      |            | LSTM  | 1.52  |
| 2019.6.6  |            | random_index_serial |                                      |            | LSTM  | 1.48  |
| 2019.6.7  |            | random_index_serial | n_epochs = 100                       |            | LSTM  | 1.59  |
| 2019.6.9  |            | random_index_serial | ........                             | ....       | GRU   | 1.53  |
| 2019.6.10 |            | random_index_serial | batch_size = 64 learning_rate = 5e-3 |            | LSTM  | 1.65  |
| 2019.6.12 |            | random_index_serial |                                      | train=[1,] | LSTM  | 1.54  |
| 2019.6.14 | feature_16 | random_index_serial |                                      | train=[1,] | LSTM  | 1.51  |
| 2019.6.16 | feature_16 | random_index_serial |                                      | train=[1,] | LSTM  | 1.53  |


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
