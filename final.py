from    tensorflow.examples.tutorials.mnist import input_data
import  tensorflow as tf 
import  numpy as np 
from    scipy.misc import  imresize
from    tqdm import tqdm
import  matplotlib.pyplot as plt

learning_rate = 0.01    #学习率
size = 512              #分割神经元 共512*512个神经元
delta = 0.03            #层与层的间隔
dL = 0.02               #每一层的大小 0.02m=2cm 尺寸2cm*2cm
batch_size = 32         #每次训练的图片张数
batch = 10              #输出层神经元个数
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

# _propogation() 衍射公式
def _propogation(u0, d=delta, N = size, dL = dL, lmb = 632.8e-9,theta=0.0):
    #Parameter 
    df = 1.0/dL
    k = np.pi*2.0/632.8e-9
    D= dL*dL/(N*lmb)
  
    #phase
    def phase(i,j):
        i -= N//2
        j -= N//2
        return ((i*df)*(i*df)+(j*df)*(j*df))
    ph  = np.fromfunction(phase,shape=(N,N))
    #H
    H = np.exp(1.0j*k*d)*np.exp(-1.0j*lmb*np.pi*d*ph) 
    #Result
    return tf.ifft2d(np.fft.fftshift(H)*tf.fft2d(u0)*dL*dL/(N*N))*N*N*df*df


#_propogation()的前端  
def propogation(u0,d,function=_propogation):
    return tf.map_fn(function,u0)

def make_random(shape):
    return np.random.random(size = shape)


def add_layer_amp(inputs,amp,size,delta): 
    return propogation(inputs,delta)*tf.cast(amp,dtype=tf.complex64)

#_change()将1*784 转化为512*512  
def _change(input_):
    return imresize(input_.reshape(28,28),(size,size),interp="nearest")

#_change()的前端
def change(input_):
    return np.array(list(map(_change,input_)))

# rang按比例放大或缩小十个区域
def rang(arr,shape,size=size,base = 512):
    return arr[shape[0]*size//base:shape[1]*size//base,shape[2]*size//512:shape[3]*size//512]
  
# 计算平均值
def reduce_mean(tf_):
    return tf.reduce_mean(tf_)

# 计算输出层的十个区域,元组是神经元的位置
def _ten_regions(a):
    return tf.map_fn(reduce_mean,tf.convert_to_tensor([
        rang(a,(53,153,53,153)),
        rang(a,(53,153,206,306)),
        rang(a,(53,153,359,459)),
        rang(a,(206,306,20,120)),
        rang(a,(206,306,144,244)),
        rang(a,(206,306,268,368)),
        rang(a,(206,306,392,492)),
        rang(a,(359,459,53,153)),
        rang(a,(359,459,206,306)),
        rang(a,(359,459,359,459))
    ]))

# _ten_region的前端
def ten_regions(logits):
    return tf.map_fn(_ten_regions,tf.abs(logits),dtype=tf.float32)

#保存文件
def download_text(msg,epoch,MIN=1,MAX=7,name=''):
    print("Download {}".format(name))
    if name == 'Phase':
        MIN = 0
        MAX = 2
    for i in range(MIN,MAX):
        print("{} {}:".format(name,i))
        np.savetxt("{}_Time_{}_layer_{}.txt".format(name,epoch+1,i),msg[i-1])
        print("Done")
        
def download_image(msg,epoch,MIN=1,MAX=7,name=''):
    print("Download images")
    if name == 'Phase':
        MIN = 0
        MAX = 2
    for i in range(MIN,MAX):
        print("Image {}:".format(i))
        plt.figure(dpi=650.24)
        plt.axis('off')
        plt.grid('off')
        plt.imshow(msg[i-1])
        plt.savefig("{}_Time_{}_layer_{}.pdf".format(name,epoch+1,i))
        print("Done")
        
def download_acc(acc,epoch):
    np.savetxt("Acc{}.txt".format(epoch+1),acc)
#

with tf.device("/cpu:0"):
    data_x = tf.placeholder(tf.float32,shape=(batch_size,size,size)) # 初始化
    data_y = tf.placeholder(tf.float32,shape=(batch_size,10))

    # 初始化振幅
    amp=[
        tf.Variable(make_random(shape=(size,size)),dtype=tf.float32),
        tf.Variable(make_random(shape=(size,size)),dtype=tf.float32),
        tf.Variable(make_random(shape=(size,size)),dtype=tf.float32),
        tf.Variable(make_random(shape=(size,size)),dtype=tf.float32),
        tf.Variable(make_random(shape=(size,size)),dtype=tf.float32),
        tf.Variable(make_random(shape=(size,size)),dtype=tf.float32)
    ]

# 建立模型
layer_1 = add_layer_amp(tf.cast(data_x,dtype=tf.complex64),amp[0],size,delta*10)
layer_2 = add_layer_amp(layer_1,amp[1],size,delta)
layer_3 = add_layer_amp(layer_2,amp[2],size,delta)
layer_4 = add_layer_amp(layer_3,amp[3],size,delta)
layer_5 = add_layer_amp(layer_4,amp[4],size,delta)
output_layer = add_layer_amp(layer_5,amp[5],size,delta)
output = _propogation(output_layer,d=0.3)

# 用MSE计算误差
logits_abs = tf.square(ten_regions(tf.abs(output)))             #output是复数，先取模，在对十个区域平方
loss = tf.reduce_mean(tf.reduce_sum(tf.square(logits_abs-data_y),reduction_indices=[1]))
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
 
# 预测正确率
pre_correct = tf.equal(tf.argmax(data_y,1),tf.argmax(logits_abs,1))
accuracy = tf.reduce_mean(tf.cast(pre_correct,tf.float32))

init = tf.global_variables_initializer()
train_epochs = 20
test_epochs = 5

# 开始训练
with tf.device("/gpu:0"):
    with tf.Session() as session:
  
        session.run(init)
        total_batch = int(mnist.train.num_examples / batch_size)

        for epoch in tqdm(range(train_epochs)):
            for batch in tqdm(range(total_batch)):
                batch_x,batch_y = mnist.train.next_batch(batch_size)
                session.run(train_op,feed_dict={data_x:change(batch_x),data_y:batch_y})


            loss_,acc = session.run([loss,accuracy],feed_dict={data_x:change(batch_x),data_y:batch_y})
            print("epoch :{} loss:{:.4f} acc:{:.4f}".format(epoch+1,loss_,acc)) 

            #保存振幅文件
            msg_amp = np.array(session.run(amp)) 
            download_text(msg_amp,epoch,name='Amp')
            download_image(msg_amp,epoch,name='Amp')
