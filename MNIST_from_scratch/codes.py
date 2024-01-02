import struct
import random
import math
import time
import array
from datetime import datetime
import gc

#
# Author: Suraj Singh Bisht
#           surajsinghbisht054@gmail.com
#
# You are free to use this code as your wise.
# Please keep author name in credit section.


class TensorValue:    
    def __init__(self, num, ops= None, left=None, right=None):
        self.num = float(num)
        self.ops = ops
        self.left = left
        self.right = right
        self.grad = 0.0
        self.leaf = True
        
        if self.left or self.right:
            self.leaf = False
    
    def __add__(self, y):
        return self.__class__(self.num + getattr(y, "num", y), ops="+", left=self, right=y)
    
    def __radd__(self,  y):
        return self.__add__(y)
    
    def __sub__(self, y):   
        return self.__add__(-y)
    
    def __rsub__(self,  y):
        return self.__add__(-y)
    
    def __mul__(self, y):     
        return self.__class__(self.num * getattr(y,"num", y), ops="*", left=self, right=y)
    
    def __rmul__(self,  y):
        return self.__mul__(y)
    
    def relu(self):
        return self.__class__(max(0, self.num), ops="rl", left=self)
    
    def exp(self):
        return self.__class__(math.exp(self.num), ops="exp", left=self)

    def __neg__(self):
        return self.__mul__(-1)
    
    def __truediv__(self, y):        
        return self.__mul__(y**-1)
    
    def __rtruediv__(self,  y):
        return self.__mul__(y**-1)
    
    def __pow__(self, y):
        return self.__class__(self.num ** getattr(y, "num", y), ops="pow", left=self, right=y)
    a nice day.
    def __repr__(self):
        return f"{self.__class__.__name__}({self.num})<{id(self)}>"
    
    
    def flush_gradient(self):
        if not self.leaf:
            self.grad = 0.0
        if isinstance(self.left, self.__class__):
            self.left.flush_gradient() 
        if isinstance(self.right, self.__class__):
            self.right.flush_gradient()
    
    def backward(self):
        self.grad = 1.0
        return self.calculate_gradient_backward()
        
    def calculate_gradient_backward(self,):
        r = getattr(self.right, "num", self.right)
        l = getattr(self.left, "num", self.left)
    
        # the derivative of f(x, y) = x + y with respect to x is simply 1
        if self.ops=="+":
            self.left.grad += (self.grad * 1.0)
            if isinstance(self.right, self.__class__):
                self.right.grad += (self.grad * 1.0)
                
        # the derivative of f(a, b) = a * b with respect to 'a' is 'b'.
        elif self.ops=="*":
            self.left.grad += (r * self.grad)
                
            if isinstance(self.right, self.__class__):
                self.right.grad += (l * self.grad)
        
        elif self.ops=="rl":
            self.left.grad += (int(self.num > 0) * self.grad)
        
        elif self.ops=="exp":
            self.left.grad += (self.num * self.grad)
                
        # the derivative of f(a, b) = a^b with respect to 'a' is 'b * a^(b-1)'
        elif self.ops=="pow":
            self.left.grad += ((r * (l ** (r - 1.0))) * self.grad)
                
            if isinstance(self.right, self.__class__):
                self.right.grad += ((l * (r ** (l - 1.0))) * self.grad)
        
    
        if isinstance(self.left, self.__class__):
            self.left.calculate_gradient_backward()
            self.left.flush_gradient()
            
        if isinstance(self.right, self.__class__):
            self.right.calculate_gradient_backward()
            self.right.flush_gradient()


class Neuron:
    def __init__(self, weights, bias=1, grad=0):
        self.data = [TensorValue(i) for i in weights]
        self.bias = TensorValue(bias)
        self.wcount = len(self.data)
        self.grad = grad
    
    def get_values(self):
        return {"data":[i.num for i in self.data], "bias":self.bias.num, "wcount":self.wcount}
    
    def activation(self, val):
        return val.relu()
    
    def feed(self, arr):
        
        return self.activation(sum([weight*num for weight, num in zip(self.data, arr)]) + self.bias)
        
    def __repr__(self):
        return f"N({self.wcount}xW.)"
    
    def __iter__(self):
        return iter(self.data)
    
    def __len__(self):
        return len(self.data)

class LinearLayer:
    def __init__(self, neurons, label="Layer"):
        self.label = label
        self.neurons = neurons 
        self.ncount = len(self.neurons)
        self.wcount = len(neurons[0].data) if neurons else 0
        self.results = []
        
        
    def get_values(self):
        return {
            "label":self.label,
            "ncount":self.ncount,
            "wcount":self.wcount,
            "neurons":[neuron.get_values() for neuron in self.neurons]
        }
    
    def feed(self, arr, rcount=None, ccount=None):
        return [neuron.feed(arr) for neuron in self.neurons]
    
    def __repr__(self):
        return f"{self.label}({self.ncount}x{self.wcount})"
    
    def __len__(self):
        return self.ncount

class BasicNet:
    def __init__(self, layers, *args, **kwargs):
        self.layers = layers
        self.args = args
        self.kwargs = kwargs
        self.pre_feed_hook = None
        self.post_feed_hook = None
        
                
    def __repr__(self):
        o = ['input(*)']
        o += [repr(i) for i in self.layers]
        return ' -> '.join(o)
        
    def feed(self, arr):
        if self.pre_feed_hook:
            arr = self.pre_feed_hook(arr)
        
        for layer in self.layers:
            arr = layer.feed(arr)
        
        if self.post_feed_hook:
            arr = self.post_feed_hook(arr)
        
        return arr
    
    def get_parameters(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                for weight in neuron.data:
                    yield weight
                yield neuron.bias
    
    def softmax(self, tvals):
        exp_logits = [val.exp() for val in tvals]
        sum_exp_logits = sum(exp_logits)
        softmax_probs = [exp_logit / sum_exp_logits for exp_logit in exp_logits]
        return softmax_probs
    
    def predict(self, *args, **kwargs):
        arr = self.feed(*args, **kwargs)
        c = dict(zip(range(9), [i.num for i in arr]))
        return c, max(c, key=c.get)
    
    def get_loss(self, input_img, label):
        target = [0]*10
        # our expectations
        target[label]=1
        # prediction
        arr = self.feed(input_img)
        # calculating mse
        loss = sum([(x-y)**2 for x,y in zip(arr, target)])/len(target)
        c = dict(zip(range(9), [i.num for i in arr]))
        return loss, max(c, key=c.get)
    
    def save(self, filename):
        import json
        layers_data = [d.get_values() for d in self.layers]
        with open(filename, "w") as fp:
            json.dump(layers_data, fp)

    def load(self, filename):
        import json
        with open(filename, "r") as fp:
            dump = json.load(fp)
            self.layers = []
            for layer in dump:
                l = LinearLayer([])
                l.label = layer['label']
                l.ncount = layer['ncount']
                l.wcount = layer['wcount']
                l.neurons = []
                for ndata in layer['neurons']:
                    o = Neuron([])
                    o.data = [TensorValue(i) for i in ndata['data']]
                    o.bias = TensorValue(ndata['bias'])
                    o.wcount = ndata['wcount']
                    l.neurons.append(o)

                self.layers.append(l)



# Dataset : https://www.kaggle.com/code/hojjatk/read-mnist-dataset/notebook
class DataHolder:
    def __init__(self):
        self.image_dimension = (28, 28)
        self.train_img = self.extract_images("Dataset/train-images.idx3-ubyte")
        self.train_label = self.extract_labels("Dataset/train-labels.idx1-ubyte")
        self.test_img = self.extract_images("Dataset/t10k-images.idx3-ubyte")
        self.test_label = self.extract_labels("Dataset/t10k-labels.idx1-ubyte")
        
    def extract_labels(self, path):
        res = []
        with open(path, 'rb') as fp:
            # extract header
            _, size = struct.unpack("!II", fp.read(4*2))
            res = array.array("B", fp.read(size))
        return res
    
    def extract_images(self, path):
        res = []
        with open(path, 'rb') as fp:
            magic_code, size, rows, cols = struct.unpack("!IIII", fp.read(4*4))
            dim = [size, rows, cols]
            buffer_count = cols*rows
            image_data = array.array("B", fp.read())
            for index in range(size):
                raw_arr = image_data[index*buffer_count:(index*buffer_count)+buffer_count]
                res.append(raw_arr)
        return res
                
        
    def get_set(self, index, test=False):
        if test:
            return self.test_img[index].tolist(), self.test_label[index]
        
        return self.train_img[index].tolist(), self.train_label[index]
    
    def get_img(self, *args, **kwargs):
        img_arr, label = self.get_set(*args, **kwargs)
        image = []
        for w in range(28):
            image.append(img_arr[w*28:(w+1)*28])
        return image, label
    
    def prev_img(self, *args, **kwargs):
        image, label = self.get_img(*args, **kwargs)
        plt.imshow(image, cmap='gray')
        plt.axis('off') 
        plt.title(label)
        plt.show()
        return image, label
        

        
dataset_obj = DataHolder()
# 728 * 10 * 10
bnet = BasicNet([
    LinearLayer([
        Neuron(random.uniform(-0.1, 0.1) for _ in range(728)) for _ in range(10)
    ], label="hidden"),
    
    LinearLayer([
        Neuron(random.uniform(-0.1, 0.1) for _ in range(10)) for _ in range(10)
    ], label="output")
    
])
bnet.pre_feed_hook = lambda arr: [v/255 for v in arr]
bnet.post_feed_hook = bnet.softmax
print(f"Trainer Started at {datetime.now()}")

bnet.load("t1.wt")
range_numbers = list(range(59990))
random.shuffle(range_numbers)

for iternum, datasetIndex in enumerate(range_numbers):
    learning_rate = random.randint(80, 150) * 0.01
    im, ll = dataset_obj.get_set(datasetIndex)
    (pre_loss, predict), actual = bnet.get_loss(im, ll), ll
    if int(predict)==int(actual):
        print(f"{datetime.now()}; {iternum}; DataIndex:{datasetIndex}; PreLoss:{round(pre_loss.num, 8)}; Prediction:{predict}; Actual:{actual}; Pass;")
        continue
    # calculating backward gradient
    pre_loss.backward()
    for w in bnet.get_parameters():
        w.num += (w.grad * learning_rate * -1)
        w.grad = 0 # set zero
    (loss, predict), _ = bnet.get_loss(im, ll), ll
    t = datetime.now()
    print(f"{datetime.now()}; {iternum}; DataIndex:{datasetIndex}; PreLoss:{round(pre_loss.num, 8)}; NowLoss:{round(loss.num, 8)}; Prediction:{predict}; Actual:{actual}; Rate:{learning_rate};")
    bnet.save(f"{t.date()}_{t.time().hour}")
    bnet.save("t1.wt")
    gc.collect()
