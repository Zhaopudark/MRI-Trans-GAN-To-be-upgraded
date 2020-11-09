import tensorflow as tf 
class Baba():
    def __init__(self):
        self.have_ass=False
    def __call__(self):
        self.in_shape=0
    @property
    def input_shape(self):
        return None
class MyDenseLayer(Baba):
    def __init__(self, num_outputs,input_shape=None):
        super(MyDenseLayer, self).__init__()
        self.input_shape_zp = input_shape
        self.num_outputs = num_outputs
    def call(self,input):
        return 0
a = MyDenseLayer(10,[None,100])
print(MyDenseLayer.input_shape)

print(a.input_shape)
print(a.input_shape_zp)
a = [1,2,3.5,None,True,False]
print(a)