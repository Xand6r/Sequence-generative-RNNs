

```python
###### using an rnn to generate signals to mimic a simple pattern #############33
###### tthis could be applied to the foreign exchange market, stocks market and basically anything with a trend i.e time series###
```


```python
# Importing every library we are going to need
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")
%matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split
```


```python
#set all my parameters
n_steps=10
n_inputs=1
n_neurons=100
n_outputs=1

#this function helps me track my models closeness to finish, i hate looking at empty screens waiting til god-know-when the model will finish training
def percent(rec,exp):
    import sys
    sys.stdout.write(str(round((rec/exp)*100,2))+"%")
    sys.stdout.write("\r")
```


```python
#run a tensorflw graph using an rnn cell

rnn=tf.Graph()
with rnn.as_default():
    x=tf.placeholder(tf.float32,[None,n_steps,n_inputs])
    y=tf.placeholder(tf.float32,[None,n_steps,n_outputs])
    # cell=tf.contrib.rnn.BasicRNNCell(num_units=n_neurons,activation_fn=tf.nn.relu)
    
#using an rnn projection wrapper so that we can have only one output per step

    cell=tf.contrib.rnn.OutputProjectionWrapper(
        tf.contrib.rnn.BasicRNNCell(num_units=n_neurons,activation=tf.nn.relu),
        output_size=n_outputs
    )

    outputs,states=tf.nn.dynamic_rnn(cell,x,dtype=tf.float32)
    learning_rate=0.0001
    
    loss=tf.reduce_mean(tf.square(outputs-y))
    optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
    
    
    training_op=optimizer.minimize(loss)
    saver=tf.train.Saver()
    init=tf.global_variables_initializer()
```


```python
# generate sequential data for testing, this is quite the easy data to generate, but we do so using a sin function
def function(t):
    import math
    return (t*math.sin(t))/3+2*math.sin(t*5)

x_data=np.arange(0,30.1,0.1)
y_data=[function(i) for i in x_data]

x=np.array([first for first,_ in zip(y_data,y_data[1:])])
y=np.array([second for _,second in zip(y_data,y_data[1:])])

x_train,x_test,y_train,y_test=x[:-10].reshape(-1,10,1),x[-10:].reshape(-1,10,1),y[:-10].reshape(-1,10,1),y[-10:].reshape(-1,10,1)
```


```python
x_train[0:15].shape
```




    (15, 10, 1)




```python
#train our data on the test model, let us see how well it can replicate it

with rnn.as_default():
    n_iterations=10000
    batch_size=5
    n=x_train.shape[0]
    with tf.Session() as sess:
        init.run()
        for iteration in range(n_iterations):
            pos=0
            while pos<n:
                percent(pos,n)
                x_batch,y_batch=x_train[pos:pos+batch_size],y_train[pos:pos+batch_size]
                sess.run([training_op],feed_dict={x:x_batch,y:y_batch})
                pos+=batch_size
            if iteration%1000==0:
                mse=loss.eval(feed_dict={x:x_batch,y:y_batch})
                print("iteration:{},mse:{}".format(iteration,mse))
        saver.save(sess,r"/temp/tf_models/rnet.ckpt")
```

    iteration:0,mse:37.2623176574707
    iteration:1000,mse:0.19994871318340302
    iteration:1000,mse:0.0001112121212122
    100.00%



```python
# restoring my sessions for when my models crash, and using it to predict
with rnn.as_default():
    with tf.Session() as sess:
        saver.restore(sess,r"/temp/tf_models/rnet.ckpt")
        predictions=sess.run(outputs,feed_dict={x:x_test})
```

    INFO:tensorflow:Restoring parameters from /temp/tf_models/rnet.ckpt
    


```python
predictions=predictions.reshape(-1,10)[0]
predictions
```




    array([ -5.52529573,  -5.34636068,  -6.17288589,  -7.73477793,
            -8.67946053,  -9.55539131, -11.03153324, -11.71816826,
           -11.70035458, -11.22727203], dtype=float32)




```python
#plot my model's predictions along with the original data to measure deviation.
plt.scatter(x_data[-11:-1],x_test.reshape(-1,10),marker='o',s=60)
plt.scatter(x_data[-10:],predictions,marker="*",color="red")
```




    <matplotlib.collections.PathCollection at 0x2646936eda0>




![png](output_9_1.png)

#### we can see the model's prediction against our own we can see that they are fairly close, this is how we use rnn to generate a sequence
