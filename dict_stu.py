from collections import OrderedDict,defaultdict
import copy,pdb
from itertools import groupby
from music21 import *
def _is_scale_tone(chord,note):
    scaleType=scale.DorianScale()
    
    
