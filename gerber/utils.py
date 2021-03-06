#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2014 Hamilton Kibbe <ham@hamiltonkib.be>

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
gerber.utils
============
**Gerber and Excellon file handling utilities**

This module provides utility functions for working with Gerber and Excellon
files.
"""

# Author: Hamilton Kibbe <ham@hamiltonkib.be>
# License:


def parse_gerber_value(value, format=(2, 5), zero_suppression='trailing'):
    """ Convert gerber/excellon formatted string to floating-point number

    .. note::
        Format and zero suppression are configurable. Note that the Excellon
        and Gerber formats use opposite terminology with respect to leading
        and trailing zeros. The Gerber format specifies which zeros are
        suppressed, while the Excellon format specifies which zeros are
        included. This function uses the Gerber-file convention, so an
        Excellon file in LZ (leading zeros) mode would use
        `zero_suppression='trailing'`


    Parameters
    ----------
    value : string
        A Gerber/Excellon-formatted string representing a numerical value.

    format :  tuple (int,int)
        Gerber/Excellon precision format expressed as a tuple containing:
        (number of integer-part digits, number of decimal-part digits)

    zero_suppression : string
        Zero-suppression mode. May be 'leading' or 'trailing'

    Returns
    -------
    value : float
        The specified value as a floating-point number.

    """
    # Handle excellon edge case with explicit decimal. "That was easy!"
    if '.' in value:
        return float(value)

    # Format precision
    integer_digits, decimal_digits = format
    MAX_DIGITS = integer_digits + decimal_digits

    # Absolute maximum number of digits supported. This will handle up to
    # 6:7 format, which is somewhat supported, even though the gerber spec
    # only allows up to 6:6
    if MAX_DIGITS > 13 or integer_digits > 6 or decimal_digits > 7:
        raise ValueError('Parser only supports precision up to 6:7 format')

    # Remove extraneous information
    #value = value.strip()
    value = value.lstrip('+')
    negative = '-' in value
    if negative:
        value = value.lstrip('-')


    digits = list('0' * MAX_DIGITS)
    offset = 0 if zero_suppression == 'trailing' else (MAX_DIGITS - len(value))
    for i, digit in enumerate(value):
        digits[i + offset] = digit

    result = float(''.join(digits[:integer_digits] + ['.'] + digits[integer_digits:]))
    return -result if negative else result


def write_gerber_value(value, format=(2, 5), zero_suppression='trailing'):
    """ Convert a floating point number to a Gerber/Excellon-formatted string.

    .. note::
        Format and zero suppression are configurable. Note that the Excellon
        and Gerber formats use opposite terminology with respect to leading
        and trailing zeros. The Gerber format specifies which zeros are
        suppressed, while the Excellon format specifies which zeros are
        included. This function uses the Gerber-file convention, so an
        Excellon file in LZ (leading zeros) mode would use
        `zero_suppression='trailing'`

    Parameters
    ----------
    value : float
        A floating point value.

    format :  tuple (n=2)
        Gerber/Excellon precision format expressed as a tuple containing:
        (number of integer-part digits, number of decimal-part digits)

    zero_suppression : string
        Zero-suppression mode. May be 'leading' or 'trailing'

    Returns
    -------
    value : string
        The specified value as a Gerber/Excellon-formatted string.
    """
    # Format precision
    integer_digits, decimal_digits = format
    MAX_DIGITS = integer_digits + decimal_digits

    if MAX_DIGITS > 13 or integer_digits > 6 or decimal_digits > 7:
        raise ValueError('Parser only supports precision up to 6:7 format')

    # Edge case... (per Gerber spec we should return 0 in all cases, see page 77)
    if value == 0:
        return '0'

    # negative sign affects padding, so deal with it at the end...
    negative = value < 0.0
    if negative:
        value = -1.0 * value

    # Format string for padding out in both directions
    fmtstring = '%%0%d.0%df' % (MAX_DIGITS + 1, decimal_digits)
    digits = [val for val in fmtstring % value if val != '.']
    
    # If all the digits are 0, return '0'.
    digit_sum = reduce(lambda x,y:x+int(y), digits, 0)
    if digit_sum == 0:
        return '0'

    # Suppression...
    if zero_suppression == 'trailing':
        while digits[-1] == '0':
            digits.pop()
    else:
        while digits[0] == '0':
            digits.pop(0)

    return ''.join(digits) if not negative else ''.join(['-'] + digits)


def decimal_string(value, precision=6, padding=False):
    """ Convert float to string with limited precision

    Parameters
    ----------
    value : float
        A floating point value.

    precision :
        Maximum number of decimal places to print

    Returns
    -------
    value : string
        The specified value as a  string.

    """
    floatstr = '%0.10g' % value
    integer = None
    decimal = None
    if '.' in floatstr:
        integer, decimal = floatstr.split('.')
    elif ',' in floatstr:
        integer, decimal = floatstr.split(',')
    else:
        integer, decimal = floatstr, "0"

    if len(decimal) > precision:
        decimal = decimal[:precision]
    elif padding:
        decimal = decimal + (precision - len(decimal)) * '0'

    if integer or decimal:
        return ''.join([integer, '.', decimal])
    else:
        return int(floatstr)


def detect_file_format(filename):
    """ Determine format of a file

    Parameters
    ----------
    filename : string
        Filename of the file to read.

    Returns
    -------
    format : string
        File format. either 'excellon' or 'rs274x'
    """

    # Read the first 20 lines
    with open(filename, 'r') as f:
        lines = [next(f) for x in xrange(20)]

    # Look for
    for line in lines:
        if 'M48' in line:
            return 'excellon'
        elif '%FS' in line:
            return'rs274x'
    return 'unknown'
