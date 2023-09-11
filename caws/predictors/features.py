from enum import Enum

class CawsFeatureType(IntEnum):
    CATEGORICAL: str = "categorical"
    CONTINUOUS: str = "continuous"

def ArgFuncFeature(feature_func, arg_no = 0, arg_name = None):
    if arg_name is not None:
        def f(*args, **kwargs):
            return feature_func(kwargs[arg_name])
        return f
    
    def f(*args, **kwargs):
        return feature_func(args[arg_no])
    return f

def _identity(x):
    return x

def ArgFeature(arg_no = 0, arg_name = None):
    return ArgFuncFeature(_identity, arg_no, arg_name)
    
def ArgAttrFeature(attr_name, arg_no = 0, arg_name = None):
    return ArgFuncFeature(lambda x : getattr(x, attr_name), arg_no, arg_name)

def ArgSizeFeature(arg_no = 0, arg_name = None):
    import sys
    return ArgFuncFeature(sys.getsizeof, arg_no, arg_name)

def LenFeature(arg_no = 0, arg_name = None):
    return ArgFuncFeature(len, arg_no, arg_name)