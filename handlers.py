

import logging

def errorHandler(func):
    def run(*args,**kwargs):
        try:
            return func(*args,**kwargs)
        #except BadParameter as e:
        #    return Error_response(msg=e.msg)
        except Exception as e:
            raise e
    return run



