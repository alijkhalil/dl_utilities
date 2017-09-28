
def mama(parm1, parm2, parm3, lala1=20, lala2=30, lala3=50):
    print(parm1, parm2, parm3)
    print(lala1, lala2, lala3)

def print_keyword_args(haha, haha2, haha3=3, **kwargs):
    #kwargs is a dict of the keyword args passed to the function
    for key, value in kwargs.iteritems():
        print "%s = %s" % (key, value)    
    print("Done.")
    
    mama(1, haha, haha2, lala2=2, **kwargs)
    
if __name__ == '__main__':
    kwargs = {'haha2': 'Bobby2', 'lala1': 'Bobby', 'lala3': 'Smith'}
    #kwargs = {'lala3': 'Bobby', 'lala4': 'Smith'}
    kwargs = {}
    print_keyword_args(100, 10, haha3=100)
    
    exit(0)