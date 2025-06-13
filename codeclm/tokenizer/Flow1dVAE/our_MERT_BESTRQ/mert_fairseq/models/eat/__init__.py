try:
    from .EAT_pretraining import *
except:
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.'))
    from EAT_pretraining import *