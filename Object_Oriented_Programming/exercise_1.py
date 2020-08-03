# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 02:17:35 2020

@author: dcamp
"""
import random 

# Note: when we call a method on an instance, the instance gets passed as an argument automatically

# Simple class (code_1)
class Person(object):
    def speak(self):
        print('Hello!')
        print('the current instance is identified as:')
        print(self)
    
    # Random number is assigned as a variable in the instance.
    def random_call(self):    
        self.rand_val = random.randint(1,10)
        
this_person = Person()
this_person.speak()

# 'self' is the instance upon the class was called
# Note: Instances := Objects (Both terms refer to the same thing)

this_person.random_call()
print(this_person.rand_val)
print('\t')
print('\t')

# Three main pillars of OOP
# 1. Encapsulation: It assures the integrity of the data inside an object by providing setter methods that help defining the desidered properties of instances
# 2. Inheritance
# 3. Polymorphism

# Example of encapsulation (code_2)

class MyInteger(object):
    def set_val(self,val):
        try:
            val = int(val)
        except ValueError:
            return 
        
        self.val = val
        
    def get_val(self):
        return self.val
    
    def increment_val(self):
        self.val += 1 

one_ex_1 = MyInteger()

# Invalid case in which 'val' is set outside and it is admitted
one_ex_1.val = 'one'


one_ex_2 = MyInteger()
# Invalid case of inserting 'one' where it gets recognized as an unvalid value right the way and it does not get assigned as a variable thanks to the encapsulation
one_ex_2.set_val('one')
print('\t')
print('\t')


# Vairiables that are set right before an instance is even created are added through the __init__ method
# The init method is a magic or provate method, which is defined automaticallt any time an instance is created
# Example constructor (code_3)
class MyNum(object):
    count = 0
    # This method is run by python automatically (constructor)
    def __init__(self,value):
        print('calling __init__')
        # Integrity gate
        try:
            value = int(value)
            MyNum.count += 1 # It is a class attribute that increases as each instance gets created
            # self.count += 1 # it is an instance atribute, it will always be either 0 or 1
        except ValueError:
            value = 0 
        self.val = value
        
    def increment(self):
        self.val += 1
    
    
num_instance_1 = MyNum(2)
num_instance_1.increment()
num_instance_1.increment()

print(num_instance_1.val)
print(num_instance_1.count)


num_instance_2 = MyNum('5')
num_instance_2.increment()
num_instance_2.increment()


print(num_instance_2.val)
print(num_instance_2.count)

# Class attributes (they are callable from the instances as shown before)
print(MyNum.count)
# When an atribute is requested, python gives priority to instance-attributes and then class-attributes

print('\t')
print('\t')


########## First assignment task (code_4) ############
class MaxSizeList(object):
    num_instances = 0 # Class data
    def __init__(self,dim):
        try:
            dim = int(dim)
        except ValueError:
            print('The object could not be created, provide an integer as input')
            return 
        MaxSizeList.num_instances += 1
        self.max_dim = dim
        self.list = []
    
    def push(self,string_val):
        try:
            string_val = str(string_val)
        except ValueError:
            print('The object could not be created, provide an string as input')
            return 
        
        if len(self.list) < self.max_dim:
            self.list.append(string_val)
        else:
            self.list.pop(0)
            self.list.append(string_val)
            
        ''' # More elegant solution (provided by the course)
        self.list.append(string_val)
        if len(self.list) > self.max_dim:
            self.list.pop(0)
        '''
    
    def get_list(self):
        return self.list
    
a = MaxSizeList(3)
b = MaxSizeList(1)

a.push('hey')
a.push('hi')
a.push('let us')
a.push('go')

b.push('hey')
b.push('hi')
b.push('let us')
b.push('go')

print(a.get_list())

print(b.get_list())

