# -*- coding: utf-8 -*-
"""
Generation of a manual NN that performs add and multiplication operations through OOP

@author: dcamp
"""
import numpy as np

class Operation():
    def __init__(self, input_nodes=[]):
        self.input_nodes = input_nodes
        self.output_nodes = []
        
        for node in input_nodes:
            node.output_nodes.append(self)
        
        _default_graph.operation.append(self)
    
    def compute(self):
        pass

# The first letters of the following classes should have been capitalized but tensorflow does not do it
class add(Operation):
    def __init__(self,x,y):
        super().__init__([x,y])
    
    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return x_var + y_var

class multiply(Operation):
    def __init__(self,x,y):
        super().__init__([x,y])
    
    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return x_var * y_var
    
class matmul(Operation):
    def __init__(self,x,y):
        super().__init__([x,y])
    
    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        # assuming x_var is a np
        return x_var.dot(y_var)
    
class Placeholder():
    def __init__(self):
        self.output_nodes = []
        # We will append the placeholder to the graph when creating it
        _default_graph.placeholders.append(self)

class Variable():
    def __init__(self,initial_value=None):
        self.value = initial_value
        self.output_nodes = []
        
        _default_graph.variables.append(self)

class Graph():
    def __init__(self):
        self.operation = []
        self.placeholders = []
        self.variables = []
    
    def set_as_default(self):
        # Using global I make the graph information accesible to all the other classes
        global _default_graph
        _default_graph = self
        
#z = Ax + b
#A = 10
#b = 1
#z = 10x + 1 
# In the case above 'x' could be whatever number, then 'x' is a placeholder

g = Graph()
g.set_as_default()
A = Variable(10)
b = Variable(1)
x = Placeholder()

y = multiply(A, x)
z = add(y, b)

def traverse_postorder(operation):
    """ 
    PostOrder Traversal of Nodes. Basically makes sure computations are done in 
    the correct order (Ax first , then Ax + b). Feel free to copy and paste this code.
    It is not super important for understanding the basic fundamentals of deep learning.
    """
    
    nodes_postorder = []
    def recurse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)

    recurse(operation)
    return nodes_postorder

class Session:
    
    def run(self, operation, feed_dict = {}):
        """ 
          operation: The operation to compute
          feed_dict: Dictionary mapping placeholders to input values (the data)  
        """
        
        # Puts nodes in correct order
        nodes_postorder = traverse_postorder(operation)
        
        for node in nodes_postorder:

            if type(node) == Placeholder:
                
                node.output = feed_dict[node]
                
            elif type(node) == Variable:
                
                node.output = node.value
                
            else: # Operation
                
                node.inputs = [input_node.output for input_node in node.input_nodes]

                 
                node.output = node.compute(*node.inputs)
                
            # Convert lists to numpy arrays
            if type(node.output) == list:
                node.output = np.array(node.output)
        
        # Return the requested node value
        return operation.output

sess = Session()
result = sess.run(operation=z,feed_dict={x:10})
print(result)

g = Graph()
g.set_as_default()

A = Variable([[10,20],[30,40]])
b = Variable([1,1])

x = Placeholder()

y = matmul(A,x)

z = add(y,b)

sess = Session()

result = sess.run(operation=z,feed_dict={x:10})
print(result)

import matplotlib.pyplot as plt
def sigmoid(x):
    return 1/(1 + np.exp(-x))

x = np.linspace(-7.0, 7.0, 100)
y = sigmoid(x)
plt.plot(x,y)
plt.show()

class Sigmoid(Operation):
    def __init__(self,z):
        super().__init__([z])
    
    def compute(self,x_val):
        return 1/(1 + np.exp(-x_val))

# Create a binary classification problem based on two features
from sklearn.datasets import make_blobs
data = make_blobs(n_samples=50, n_features=2, centers=2, random_state=75)
# data is a tuple of dimension 2. The first dimensions contains feature information whereas the second dimension contains label information 
features = data[0]
labels = data[1]
plt.scatter(features[:,0], features[:,1], c=labels, cmap='coolwarm' )
x = np.linspace(-1.0,10, 10)
y = -x + 4.5
plt.plot(x,y)


# We can write the decision boundary (of this particular case) for classification in the following form
# (1,1).*features - 4.5 = 0 ----> on the boundary (not class 1 nor class 2)
# (1,1).*features - 4.5 > 0 ----> class 1
# (1,1).*features - 4.5 < 0 ----> class 2
# Since y + x - 4.5 = 0

class_sel = np.sign(np.array([1,1]).dot(np.array([0,-5])) - 4.5)
plt.scatter(0,-5,c='black')
print(f"class(black): {class_sel}")
class_sel = np.sign(np.array([1,1]).dot(np.array([5,8])) - 4.5)
plt.scatter(5,8,c='green')
print(f"class(green): {class_sel}")

plt.show()

g = Graph()
g.set_as_default()
x = Placeholder()
w = Variable([1,1])
b = Variable(-5)
z = add(matmul(w,x),b)
a = Sigmoid(z)
sess = Session()
result_1 = sess.run(operation=a,feed_dict={x:[0,-5]})
result_2 = sess.run(operation=a,feed_dict={x:[5,8]})
print(f'By using the sigmoid function, black dot: {result_1}')
print(f'By using the sigmoid function, geen dot: {result_2}')