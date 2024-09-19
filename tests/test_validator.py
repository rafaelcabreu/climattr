import pytest

from climattr.validator import (
    validate_ci,
    validate_correction_method,
    validate_direction
)

def test_validate_ci_valid():
    # Test valid CI values
    validate_ci(1)
    validate_ci(50)
    validate_ci(100)

###############################################################################

def test_validate_ci_invalid():
    # Test invalid CI values
    with pytest.raises(ValueError):
        validate_ci(0)  # Below range
    with pytest.raises(ValueError):
        validate_ci(101)  # Above range
    with pytest.raises(ValueError):
        validate_ci('50')  # Not an integer
    with pytest.raises(ValueError):
        validate_ci(50.5)  # Not an integer

###############################################################################

def test_validate_direction_valid():
    # Test valid direction values
    validate_direction('ascending')
    validate_direction('descending')

###############################################################################

def test_validate_direction_invalid():
    # Test invalid direction values
    with pytest.raises(ValueError):
        validate_direction('up')
    with pytest.raises(ValueError):
        validate_direction('down')
    with pytest.raises(ValueError):
        validate_direction('ascend')
    with pytest.raises(ValueError):
        validate_direction(123)  # Not a string

###############################################################################

def test_validate_correction_method_valid():
    # Test valid correction method values
    validate_correction_method('add')
    validate_correction_method('mult')

###############################################################################

def test_validate_correction_method_invalid():
    # Test invalid correction method values
    with pytest.raises(ValueError):
        validate_correction_method('subtract')
    with pytest.raises(ValueError):
        validate_correction_method('multiply')
    with pytest.raises(ValueError):
        validate_correction_method('divide')
    with pytest.raises(ValueError):
        validate_correction_method(123)  # Not a string

###############################################################################
