'''
IMPORTANT NOTE:
For testing myprogram.py, go to the terminal at the directory in question and run the command 'pytest test_with_myprogram.py'
'''

# In a file called test_myprogram.py  (The prefix test_ is neccesary in for testing the code)
###############################################    
# We are not going to run "myprogram", all we are gonna do it is import it to have access to its functions and test them individually

import pytest
import myprogram 

# The prefix "test_" is neccesary here for making python understand we are taking the doubleit method from myprogram.py
def test_doubleit():
    # assert remains silent if the argument at its right is True 
    assert myprogram.doubleit(10) == 20
    
def test_doubleit_type():
    # This is a away of saying that 'hello' will arise a TypeError
    # In other words, we are assuming that hellp will produce a Type error 
    with pytest.raises(TypeError):
        myprogram.doubleit('hello')