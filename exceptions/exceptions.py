class BadParameter(Exception):
    """Custom exception for bad parameters (input)"""
    
    def __init__(self,msg,*args,**kwargs):
        self.msg=msg

    def __str__(self):
        print(self.msg)