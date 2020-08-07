'''
Use the bult-in class dict in order to make a class that allows us 
to read key value pairs from a .txt file and write/modify them in 
the .txt file in question (test_exercise_3.txt)
'''

# Proposed solution 
class ConfigDict(dict):
    
    def __init__(self, filename):
        try: 
            file_ = open(filename,'r')
        except:
            raise ImportError('The file provided cannot be opened!')
        
        # private attributes start with an underscore
        self._filename = filename
        all_lines = file_.readlines()
        keys = []
        for line in all_lines:
            if '=' in line:
                key, value = line.split('=',1)
                dict.__setitem__(self, key, value)
                keys.append(key)
        self.keys = keys
        if all_lines[-1] != '\n':
            all_lines[-1] = all_lines[-1] + '\n'
        self.all_lines = all_lines
        file_.close()

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)
        if key in self.keys:
            ind_change = self.keys.index(key)
            self.all_lines[ind_change] = '{0}={1}\n'.format(key, value)
        else:
            self.all_lines.append('{0}={1}\n'.format(key, value))
            self.keys.append(key)
        
        with open(self._filename,'w') as file_:
                file_.writelines(self.all_lines)


# Course solution (imcomplete since it overwrites information in the .txt file instead of adding new key-value pair information)
import os

class ConfigDictCourse(dict):
    
    def __init__(self, filename):
        
        self._filename = filename
        if os.path.isfile(self._filename):
            with open(self._filename) as fh:
                for line in fh:
                    line = line.rstrip()
                    key, value = line.split('=',1)
                    dict.__setitem__(self, key, value)
                    
    def __setitem__(self, key, value):
        
        dict.__setitem__(self, key, value)
        with open(self._filename, 'w') as fh:
            for key, val in self.items():
                fh.write('{0}={1}\n'.format(key, val))

            
filename = 'test_exercise_3.txt'
dict_ = ConfigDict(filename)
# dict_ = ConfigDictCourse(filename)