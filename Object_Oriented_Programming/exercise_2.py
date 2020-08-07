# Polimorphism refers to the idea that different calsses may have the same method name.
# Although the method may do similar things on different classes, generally, they are written differently such that they are applicable to the class in question. 

# As classes become more complex, we may whish to initialize an instance by first processing it in the parent-class constructor and then through the child-class constructor

import random 

class Animal(object):
    
    def __init__(self, name):
        self.name = name 

# Dog inherits the properties (attributes of Animal)
class Dog(Animal):
    
    # Super is used to refer to the superclass (parent class) and call its constructor inside a child class
    # Optionally, the ParentClss.__init__(self,args) can be also used and it is actually more general since it allows us to call the constructors of multiplle parent classes. (This is called multiple inheritance) 
    def __init__(self, name):
        # super(Dog,self).__init__(name)
        Animal.__init__(self, name)
        self.breed = random.choice(['Shih Tzu', 'Beagle', 'Dalmata', 'German Shepard'])
        
    def fetch(self, thing):
        print ('%s goes after the %s!' % (self.name, thing))

d = Dog('Snoop Dogg')

print(d.name)
print(d.breed)



# In the following exercise GetterSetter is defined as an abstract class that works as a blueprint for class 'My class'
# Note that GetterSetter that the GetterSetter does not generate instances but define the characteristics of a class

import abc 
class GetterSetter(object):
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def set_val(self, input):
        ''' set a value in the instance'''
        return 
    
    @abc.abstractmethod
    def get_val(self):
        '''Get and return value from the instance'''
        return 
    
class MyClass(GetterSetter):
    
    def set_val(self, input):
        self.val = input
        
    def get_val(self):
        return self.val
    
x = MyClass()
x.set_val(1)
print(x.get_val())
print('MyClass is of the type: ' + str(type(MyClass)))


#################################################################################################################
# Class that allows summing the numerical content of two lists (without using other packages such as numpy or pandas)
class SumList(object): # class SumList(list): would inherit the all list's methods
    
    def __init__(self, this_list):
        if type(this_list) != list :
            raise TypeError('The input must be a list')
        self.this_list = this_list
    
    def __add__(self, sum_list_instance):
        
        if len(self.this_list) != len(sum_list_instance.this_list):
            raise ValueError('Both lists should be equal')
            
        # List conprehession that sums both objects component by component 
        summed_list = [x+y for x,y in zip(self.this_list, sum_list_instance.this_list)]
        return SumList(summed_list)

list_a = SumList([1,2,3,4,5,6,7,8,9])
list_b = SumList([10,20,30,40,50,60,70,80, 90])

list_ab = list_a + list_b
print(list_ab.this_list)

###################################################################################

'''
More complex exercise: Use the inheritance property of classes for creating two subclasses that allow to write data to .txt and .csv formats
'''

# Proposed solution (it checks for the different types of data required for the application and allows writing ',' symbols in the .csv file)
import datetime

class WriteFile(object):
    def __init__(self, name_file):
        self.name_file = name_file
        
    def write(self, text_input):
        self.text_input = text_input
        file_write = open(self.name_file,'w')
        file_write.write(self.text_input + '\n')
        file_write.close()
        print(self.text_input)

    @staticmethod
    def list_verify(text_input):
        if (type(text_input) != list):
            raise TypeError('The input must be a list')
        return text_input
    
    @staticmethod
    def str_verify(text_input):
        if (type(text_input) != str):
            raise TypeError('The input must be a list')
        return text_input 
    
class LogFile(WriteFile):
    
    def write(self, text_input):
        text_input = WriteFile.str_verify(text_input)
        text_input = datetime.datetime.now().strftime('%Y-%m-%d %H:%M') + ' '*4 + text_input
        WriteFile.write(self,text_input)
        
class DeLimFile(WriteFile):
    def __init__(self, name_file, delimeter):
        WriteFile.__init__(self, name_file)
        self.delimeter = ','
        
    def write(self, text_input):
        text_input = WriteFile.list_verify(text_input)
        output = ''
        text_input_final = []
        for i in range(len(text_input)):
            if ',' in text_input[i]:
                text_input[i] = '"{0}"'.format(text_input[i])
            text_input_final.append(text_input[i])
        
        output = self.delimeter.join(text_input_final)
        WriteFile.write(self,output)


log = LogFile('log.txt')
c = DeLimFile('text.csv', ',')

c.write(['a', 'b', 'c', 'd'])
c.write(['1', '2,', '3', '4'])

log.write('this is a log message')
log.write('this is another log message')

################################################################################################################

# Solution of the assignment (Cleaner and simpler version of the algorithm with basic functionalities)
import abc

class WriteFileSol(object):
    
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def write(self):
        return 
    
    def __init__(self, filename):
        self.filename = filename
        
    def write_line(self, text):
        fh = open(self.filename, 'a')
        fh.write(text + '\n')
        fh.close()
        
class DeLimFileSol(WriteFileSol):
    
    def __init__(self,filename, delim):
        WriteFileSol.__init__(self, filename)
        self.delim = delim
        
    def write(self, this_list):
        line = self.delim.join(this_list)
        self.write_line(line)
        
class LogFileSol(WriteFileSol):
    
    def write(self, this_line):
        dt =  datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        self.write_line('{0}    {1}'.format(dt, this_line))
        
        
# log = LogFileSol('log.txt')
# c = DeLimFileSol('text.csv', ',')

# c.write(['a', 'b', 'c', 'd'])
# c.write(['1', '2,', '3', '4'])

# log.write('this is a log message')
# log.write('this is another log message')







