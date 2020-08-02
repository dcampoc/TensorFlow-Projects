# -*- coding: utf-8 -*-
"""
Object oriented programming

@author: dcamp
"""

x = 0.2
print(type(x))
print('Note that the variable x is an object (based on pre-built python class)')
print('is x an integer: {}'.format(x.is_integer()))
print('\n')
#############################################################
def hello():
    print('hello')
print(hello)
print('Note that the hello is a function')
print('\n')

print('A class is now defined'.upper())
# Use upper case to define classes
class Dog: 
    def __init__(self,name,age=0,pee_num=0):
        Dog.name = name
        Dog.pee = pee_num
        Dog.age = age
        print('My name is {} and I have peed {} times'.format(name,pee_num))
    def bark(self):
        print("guaf, guaf")
    
    def pee_(self):
        Dog.pee += 1
        
    def pee_times(self,times):
        Dog.pee = Dog.pee + times
        
    def info(self):
        print('After a while, {} has peed {} times'.format(Dog.name, Dog.pee))
        print("{}'s age is {}".format(Dog.name, Dog.age))
        
dog_1 = Dog(name='Tim',age=5)
print(type(dog_1))
dog_1.bark()
dog_1.pee_()
dog_1.pee_times(8)
dog_1.pee_()
dog_1.info()
print('\n')
#############################################################

print('Another example'.upper())

class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade
    
    def get_grade(self):
        return self.grade
    
class Course:
    def __init__(self, name, max_students):
        self.name = name 
        self.max_students = max_students
        self.students = []
    
    def add_student(self,student):
        if len(self.students) < self.max_students:
            self.students.append(student)
            return True
        return False
    
    def get_average_grade(self):
        avg_grade = []
        for s in range(0,len(self.students)):
            avg_grade.append(self.students[s].get_grade())
        return sum(avg_grade) / len(self.students)
    
    def get_average_grade_2(self):
        avg_grade = 0
        for s in self.students:
            avg_grade += s.get_grade()
        return avg_grade / len(self.students)
    
s1 = Student('Tim', 19, 70)
s2 = Student('Damian', 26, 90)
s3 = Student('Martha', 21, 85)

course_1 = Course('Biology', 2)
course_1.add_student(s1)
course_1.add_student(s2)
course_1.add_student(s3)

print(course_1.students[0].name)
print(course_1.students[1].name)
print('Note that Martha could not be added to the course')
print('\n')

##############################################################
# The parent class is deifned as Pet
class Pet:
    def __init__(self, name, age):
        print('loading info...')
        self.name = name 
        self.age = age
        
    def speak(self):
        print('I do not what to say')
    
    def show(self):
        print(f"My name is {self.name} and I am {self.age} years old")

# Child classes are defined as specific animals 
class Cat(Pet):
    def __init__(self, name, age, race='normal cat'):
        super().__init__(name, age)
        self.race = race
        
    def speak(self):
        print('meow')
    
    def race_(self):
        print(f'I am a {self.race}')

p_1 = Cat(name='Billie', age=6)
p_1.show()
p_1.speak()
p_1.race_()
print('\n')

class Fish(Pet):
    def __init__(self, name, age, color):
        super().__init__(name, age)
        self.color = color
    
    def show(self):
        print(f"My name is {self.name} and I am {self.age} years old and I am {self.color}")
p_2 = Fish('bubbles', 3, 'green')
p_2.show()
p_2.speak()
print('\n')

######################################################

print("Let's see some examples of class attributes and class methods")
class Person:
    # Number is a class attribute NOT an instance attribute (that's why it does not have self)
    number_of_people = 0
    def __init__(self,name):
        self.name = name 
        #Person.number_of_people += 1
        Person.add_person()
        
    # Include a class method,note that instead of self, it is included 'cLs'
    @classmethod
    def num_people(cLs):
        return cLs.number_of_people
    
    @classmethod
    def add_person(cLs):
        cLs.number_of_people += 1
      
    # Static methods are useful to group different functions under a single class, there is no need to create an instance to use them
    @staticmethod
    def multiply5(x):
        return x*5
    
    @staticmethod
    def info():
        print('you are using the Person class')
        
print(f"Initial number of people: {Person.number_of_people}")
p1 = Person('Tim')
p2 = Person('Damian')
print(f"Final number of people: {Person.number_of_people}")

print(Person.multiply5(6))
Person.info()


