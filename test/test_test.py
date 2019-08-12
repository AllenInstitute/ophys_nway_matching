import pytest
from nway.nway_matching_main import main, sum_me


def test_first_test():
    assert sum_me(2, 2) == 4
