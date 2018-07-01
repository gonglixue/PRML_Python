#基本使用
* 使用graph来表示计算任务
* Session：一种上下文，在其中执行图
* Variable：维护状态
* *使用feed和fetch可以为任意的操作(op)赋值或从中获取数据*

### 综述
tensor: batch, h, w, c
op: 图中的一个节点，一个op获得多个Tensor，产生多个Tensor
Session：图在会话里被启动，会话将图的op分发到device上，同时提供执行op的方法。这些方法执行后，将产生的tensor返回（在py中返回的是numpy类型）

### 计算图
数据流图用node和edge组成的有向图来描述数学运算。节点一般用来表示施加的数学操作，也可以表示数据输入的起点和输出的终点。
构建完模型所需要的图之后，还需要打开一个会话来运行整个计算图。

基本单位：
**Constant：**常量，值和维度不可变。在神经网络中可作为存储超参

**Variable：**变量，值可变，维度不可变，一般作为存储权重的矩阵
```python
import tensorflow as tf
state = tf.Variable(0, name='counter')
one = tf.constat(1) # 定义常量
# 定义一个加法op，此步骤并没有直接计算
new_value = tf.add(state, one)
# 将state更新成new_value
update = tf.assign(state, new_value)

```
定义了Variable变量，一定要初始化`init = tf.initialize_all_variables()`


**Placeholder：**没有初始值，只会分配必要的内存。使用feed_dict馈送数据。一般用于神经网络的输入。如果想要从外部传入数据到tensorflow计算图，就要用到tf.placeholder()


### 会话控制
两种形式
```python
sess = tf.Session()
result = sess.run(product)
print(result)
sess.colse()

```

```python
with tf.Session() as sess:
    result = sess.run(product)
    print(result)
```