

def validate_ci(value):

    if not (isinstance(value, int) and 0 < value <= 100):
        raise ValueError("ci must be an integer between 1 and 100.")

###############################################################################

def validate_direction(value):

    if value not in ['ascending', 'descending']:
        raise ValueError("direction must be either 'ascending' or 'descending'.")

###############################################################################

def validate_correction_method(value):

    if value not in ['add', 'divide']:
        raise ValueError("method must be either 'add' or 'divide'.")

###############################################################################
