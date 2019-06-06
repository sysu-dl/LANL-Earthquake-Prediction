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

| Date     | Features | Indexs              | Parameters | Datasets | Model | Score |
| -------- | -------- | ------------------- | ---------- | -------- | ----- | ----- |
| 2019.6.5 |          | random_index        |            |          | LSTM  | 1.52  |
| 2019.6.6 |          | random_index_serial |            |          | LSTM  | 1.48  |

