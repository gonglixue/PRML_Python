tf.estimator 包含了很多与训练好的模型，可以很快地初始化一个NN分类器。例如：
```python
feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]

#build 3 layer DNN with 10, 20, 10 units
classifier = tf.estimator.DNNClassifier(
feature_columns=feature_columns,
hidden_units=[10, 20, 10],
n_classes=3,
model_dir="temp/iris_model"
)
```

`tf.feature_column.nueric_column`：构建feature column，这里每个样本有4个feature，且都是实数值。【是定义placeholder吗？

### 描述input
`tf.estimator.inpupts.numpy_input_fn`:生成模型输入

?? train_input_fn是个function吗？怎么用
```python
train_input_fn = tf.estimator.inputs.numpy_input_fn(
x={"x": np.array(training_set.data)},
y=np.array(training_set.target),
num_epochs=None,
shuffle=True
)
```