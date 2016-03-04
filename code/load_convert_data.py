import numpy as np
from datetime import datetime

convstrlist = lambda s: datetime.strptime(s[0]+' '+s[1], '%m-%d-%Y %H:%M')
convdata = lambda l: [convstrlist(l), int(l[2]), int(l[3])]

def load_ddata(filename):
    datalist = []
    with open(filename, 'r') as infile:
        for l in infile:
            if not 'Hi' in l and not 'Lo' in l and '.'  not in l: #Throw away '0Hi'/'0Lo' values for now
                datalist += [convdata(l.split())]
    return datalist
