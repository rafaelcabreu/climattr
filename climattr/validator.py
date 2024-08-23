

def validate_ci(value):
    """
    Validate if a given confidence interval percentage is within the acceptable range.

    Parameters
    ----------
    value : int
        The confidence interval (CI) percentage to validate.

    Raises
    ------
    ValueError
        If the confidence interval is not an integer within the range of 1 to 100 
        inclusive.

    Returns
    -------
    None
        This function does not return any value; it solely performs validation.

    Notes
    -----
    This function ensures that the confidence interval used in statistical analyses
    is expressed as a valid percentage.
    """
    if not (isinstance(value, int) and 0 < value <= 100):
        raise ValueError("ci must be an integer between 1 and 100.")

###############################################################################

def validate_direction(value):
    """
    Validate if a given direction for sorting or computing is one of the acceptable 
    options.

    Parameters
    ----------
    value : str
        The direction to validate, which can be 'ascending' or 'descending'.

    Raises
    ------
    ValueError
        If the direction is not 'ascending' or 'descending'.

    Returns
    -------
    None
        This function does not return any value; it solely performs validation.
    """
    if value not in ['ascending', 'descending']:
        raise ValueError("direction must be either 'ascending' or 'descending'.")

###############################################################################

def validate_correction_method(value):
    """
    Validate if a given correction method is one of the accepted types.

    Parameters
    ----------
    value : str
        The correction method to validate, which can be 'add' for additive or 
        'mult' for multiplicative.

    Raises
    ------
    ValueError
        If the method is not 'add' or 'mult'.

    Returns
    -------
    None
        This function does not return any value; it solely performs validation.
    """
    if value not in ['add', 'mult']:
        raise ValueError("method must be either 'add' or 'mult'.")

###############################################################################
