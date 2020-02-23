from exceptions.exceptions import BadParameter

class Parser():

    def __init__(self,body):
        self.body = body
        self.green = {
            "STATUS" : (str, ["0", "1", "2", "3", "4", "5", "C", "X"]),
            "NAME_EDUCATION_TYPE": (str, ["Academic degree", "Higher education,Incomplete higher", "Lower secondary,Secondary / secondary special"]),
            "NAME_INCOME_TYPE": (str, ["Commercial associate", "Pensioner", "state servant", "Student", "Working"]),
            "CODE_GENDER": (str, ["F", "M"]),
            "NAME_FAMILY_STATUS": (str, ["Civil marriage", "Married", "Separated", "Single / not married","Widow" ]),
            "NAME_HOUSING_TYPE": (str, ["Co-op apartment", "House / apartment", "Municipal apartment" , "Office apartment", "Rented apartment", "With parent"]),
            "CNT_CHILDREN": (int, None),
            "FLAG_OWN_CAR": (bool, None),
            "FLAG_MOBIL": (bool, None),
            "FLAG_WORK_PHONE": (bool, None),
            "FLAG_PHONE": (bool, None),
            "FLAG_EMAIL": (bool, None),
            "CTM_FAM_MEMBERS":(int, None),
            "FLAG_OWN_REALTY": (bool, None),
            "DAYS_BIRTH": (int, None),
            "DAYS_EMPLOYED": (int, None),
            "MONTHS_BALANCE": (int, None),
            "AMT_INCOME_TOTAL": (float, None),
            "OCCUPATION_TYPE" : (str, ["Accountants","Cleaning staff","Cooking staff","Core staff","Drivers","HR staff","High skill tech staff", "IT staff","Laborers","Low-skill Laborers","Managers","Medicine staff","Private service staff", "Realty agents", "Sales staff","Secretaries","Security staff","Waiters/barmen staff"])
        }

    def validate(self):
        """Function that validate a body input"""

        self.body = {key:value for key,value in self.body.items() if key in self.green.keys()}
        ers = ["{} is not a {} value".format(k, self.green.get(k)[0]) for k, v in self.body.items() if not isinstance(v, self.green.get(k)[0])]
        [ers.append("{} argument missing".format(k)) for k,v in self.green.items() if k not in self.body.keys()]
        [ers.append("{} is invalid, it must be one of the following {} ".format(k,self.green.get(k)[1])) for k, v in self.body.items() if self.green.get(k)[1] and v not in self.green.get(k)[1]]

        if ers: 
            raise BadParameter(msg=", ".join(ers))



