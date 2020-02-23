




class Parser():
    def __init__(self,body):
        self.body = body
        self.greenparam = {
            STATUS : (None, [0,1,2,3,4,5,C,X])
            NAME_EDUCATION_TYPE : (str, ["Academic degree", "Higher education,Incomplete higher", "Lower secondary,Secondary / secondary special"])
            NAME_INCOME_TYPE : [Commercial associate, Pensioner,state servant ,Student,Working ]
            CODE_GENDER : [F , M]
            NAME_FAMILY_STATUS : [Civil marriage, Married , Separated, Single / not married,Widow ]
            NAME_HOUSING_TYPE : [Co-op apartment, House / apartment, Municipal apartment , Office apartment, Rented apartment, With parent]
            CNT_CHILDREN : int 0-19
            FLAG_OWN_CAR : (bool,None)
            FLAG_MOBIL : bool
            FLAG_WORK_PHONE : bool
            FLAG_PHONE : bool
            FLAG_EMAIL : bool
            CTM_FAM_MEMBERS = int 1 a 20
            FLAG_OWN_REALTY : bool
            DAYS_BIRTH : int
            DAYS_EMPLOYED : int
            MONTHS_BALANCE : int 
            AMT_INCOME_TOTAL : float
            OCCUPATION_TYPE = 
        }
         

    def validate():
        """Function that validate a body input"""

        if not self.body:
            return False

        errors = list()

        #todo esto puede ir en dos lineas
        for key, value in self.body:
            
            if not isinstance(value, type(self.greenparam.get(key)[0])):
                errors.append({"msg": "{} is not a {} value".format(key, type(self.greenparam.get(key)[0]))})

            if self.greenparam.get(key)[1]:
                if value not in self.greenparam.get(key)[1]
                    errors.append({"msg":"{} is invalid, it must be one of the following values {} ".format(key,type(self.greenparam.get(key)[1]))})
        #####
        
        if errors:
            raise BadParameter(msg=", ".join(errors.values()))

class BadParameter(Exception):
    def __init__(self,msg,*args,**kwargs):
        self.msg=None


    #def traduction():
    # For the future, token traduction