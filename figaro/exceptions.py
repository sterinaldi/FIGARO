import traceback as tb
import numpy
import sys

class FIGAROException(Exception):
    pass

def except_hook(exctype, value, traceback):
    tb_last   = list(tb.walk_tb(traceback))[-1][0] # Get last call from traceback (function that raised the exception)
    try:
        tb_s2last = list(tb.walk_tb(traceback))[-2][0] # Get second-to-last call from traceback (hopefully FIGARO function, to be checked)
    except:
        tb_s2last = ""
    # Check if error is due to some known improper usage of code
    #-----------#
    # Sample outside boundaries
    if exctype == ValueError and tb_last.f_code.co_name == "numpy.random.mtrand.RandomState.choice" and tb_s2last.f_code.co_name == "_assign_to_cluster":
        sys.__excepthook__(exctype, value, traceback)
        print("\nFIGAROException: you probably have a sample that falls outside the given boundaries\n")
    elif exctype == numpy.linalg.LinAlgError and tb_last.f_code.co_name == "_check_finite_matrix" and tb_s2last.f_code.co_name == "_log_predictive_likelihood":
        sys.__excepthook__(exctype, value, traceback)
        print("\nFIGAROException: you probably have a sample that falls outside the given boundaries\n")
    else:
        sys.__excepthook__(exctype, value, traceback)
