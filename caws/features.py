from enum import Enum
from collections import namedtuple
import inspect

class CawsFeatureType(Enum):
    CATEGORICAL: str = "categorical"
    CONTINUOUS: str = "continuous"

CawsTaskFeature = namedtuple("CawsTaskFeature", ["func", "feature_type"])

def ArgFuncFeature(feature_func, arg_no = 0, arg_name = None, feature_type=CawsFeatureType.CONTINUOUS):
    if arg_name is not None:
        def f(func, *args, **kwargs):
            bound_args = inspect.signature(func).bind_partial(*args, **kwargs)
            bound_args.apply_defaults()
            val = bound_args.arguments[arg_name]
            return feature_func(val)
    else:
        def f(func, *args, **kwargs):
            bound_args = inspect.signature(func).bind_partial(*args, **kwargs)
            bound_args.apply_defaults()
            val = list(bound_args.arguments.values())[0]
            return feature_func(val)

    return CawsTaskFeature(f, feature_type)

def _identity(x):
    return x

def ArgFeature(arg_no = 0, arg_name = None, feature_type=CawsFeatureType.CONTINUOUS):
    return ArgFuncFeature(_identity, arg_no, arg_name, feature_type)
    
def ArgAttrFeature(attr_name, arg_no = 0, arg_name = None, feature_type=CawsFeatureType.CONTINUOUS):
    return ArgFuncFeature(lambda x : getattr(x, attr_name), arg_no, arg_name, feature_type)

def ArgSizeFeature(arg_no = 0, arg_name = None, feature_type=CawsFeatureType.CONTINUOUS):
    import sys
    return ArgFuncFeature(sys.getsizeof, arg_no, arg_name, feature_type)

def ArgLenFeature(arg_no = 0, arg_name = None, feature_type=CawsFeatureType.CONTINUOUS):
    return ArgFuncFeature(len, arg_no, arg_name, feature_type)