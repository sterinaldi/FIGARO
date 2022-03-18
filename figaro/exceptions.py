import traceback as tb

def except_hook(exctype, value, traceback):
    tb_last = list(tb.walk_tb(traceback))[-1][0] # Get last call from traceback
    # Check if error is due to some known improper usage of code
    #-----------#
    # Sample outside boundaries
    if exctype == ValueError and tb_last.f_lineno == 935 and tb_last.f_code.co_filename == "mtrand.pyx":
        print("You probably have a sample that falls outside the given boundaries. Please check.")
    else:
        sys.__excepthook__(exctype, value, traceback)
