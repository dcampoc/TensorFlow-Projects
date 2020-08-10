###############################################
# In a file called myprogram.py
###############################################
import sys 

def doubleit(x):
    try:
       x = float(x)
    except:
        raise TypeError('The input should be a number')
    var = x * 2
    return var

# This gate says if we execute the script directly, we will execute these lines but if we import the program from another script, all the code is loaded but this part will not get excecuted
# In other words, this part of the code ensures that any import of the code does not execute the following lines
# Any executable lines of the the import 'myprogram' does not get executed (This is the mark of any profesionally used program in python)
if __name__ == '__main__':
    input_val = sys.argv[1]
    doubled_val = doubleit(input_val)
    
    print('the value of {0} is {1}'.format(input_val, doubled_val))