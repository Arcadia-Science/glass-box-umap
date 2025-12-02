"""An example module for demonstrating docstring conventions.

To be rendered properly in the documentation, docstrings should follow the conventions
set forth in this page. These conventions follow the
[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).
"""

CONSTANT: int = 42
"""Docstrings for variables are added directly below."""


def add_two_numbers(a: float, b: float) -> float:
    """An example function that sums two numbers.

    Args:
        a: A number.
        b: Another number.

    Returns:
        float: The sum of ``a`` and ``b``

    Examples:
        Standard usage:

        >>> example_function(42, 1)
        43
    """
    return a + b


def subtract_two_numbers(a: float, b: float) -> float:
    """An example function that subtacts two numbers.

    Args:
        a: A number.
        b: Another number.

    Returns:
        float: The subtraction of ``b`` from ``a``.

    See Also:
        - To add instead of subtract, see :func:`add_two_numbers`.
    """
    return a - b
