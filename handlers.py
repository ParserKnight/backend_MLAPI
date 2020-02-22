

import logging

def errorHandler(func):
    def run(*args,**kwargs):
        try:
            return func(*args,**kwargs)
        #except badRequest as i:
        #    return {"error_bad_request"}
        except Exception as e:
            raise e
    return run