from functools import wraps
import gc
import timeit
import logging

logger = logging.getLogger(__name__)

# Measure execution time for function
def MeasureTime(f):
    @wraps(f)
    def _wrapper(*args, **kwargs):
        gcold = gc.isenabled()
        gc.disable()
        start_time = timeit.default_timer()
        try:
            result = f(*args, **kwargs)
        finally:
            elapsed = timeit.default_timer() - start_time
            if gcold:
                gc.enable()
            print("Function '{}': {} ms".format(f.__name__, 1000.0*elapsed), flush=True)
        return result
    return _wrapper

# Put the following line outside the block to time:
#   with utils.MeasureBlockTime("MyBlock"):
class MeasureBlockTime:
    def __init__(self,name="(block)", no_print = False, disable_gc = True):
        self.name = name
        self.no_print = no_print
        self.disable_gc = disable_gc
    def __enter__(self):
        if self.disable_gc:
            self.gcold = gc.isenabled()
            gc.disable()
        self.start_time = timeit.default_timer()
    def __exit__(self,ty,val,tb):
        self.elapsed = timeit.default_timer() - self.start_time
        if self.disable_gc and self.gcold:
            gc.enable()
        if not self.no_print:
            print('Block "{}": {} ms'.format(self.name, 1000.0*self.elapsed), flush=True)
