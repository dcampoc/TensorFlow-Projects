###############################################
# In a file called myprogram.py
###############################################
import sys 

def doubleit(x):
    var = x * 2
    return var

# This gate says if we execute the script directly, we will execute these lines but if we import the program from another script, all the code is loaded but this part will not get excecuted
# In other words, this part of the code ensures that any import of the code does not execute the following lines
# Any executable lines of the the import 'myprogram' does not get executed (This is the mark of any profesionally used program in python)
if __name__ == '__main__':
    input_val = sys.argv[1]
    doubled_val = doubleit(input_val)
    
    print('the value of {0} is {1}'.format(input_val, doubled_val))


###############################################
# In a file called test_myprogram.py  (The prefix test_ is neccesary in for testing the code)
###############################################    
# We are not going to run "myprogram", all we are gonna do it is import it to have access to its functions and test them individually
import myprogram 

# The prefix "test_" is neccesary here for making python understand we are taking the doubleit method from myprogram.py
def test_doubleit():
    # assert remains silent if the argument at its right is True 
    assert myprogram.doubleit(10) == 20
    

    
    