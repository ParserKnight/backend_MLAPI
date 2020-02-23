import logging
from exceptions.exceptions import BadParameter

def errorHandler(func):
    """Handler for exceptions on endpoint"""

    def wrapper(*args,**kwargs):
        """function wrapper"""

        try:
            return func(*args,**kwargs)

        except BadParameter as e:
            #add error model
            return ({"error_msg":e.msg}), 400

        except Exception as e:
            raise e

    return wrapper



