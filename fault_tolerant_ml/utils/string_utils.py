"""All string utility functions
"""

def dict_to_str(my_dict, choose, joiner="-"):
    """Converts dictionary to string for all values in choose given a joiner

    Args:
        my_dict (dict): Dictionary to be converted to string
        joiner (str): Value that joins strings
        choose (list): List of values to include in string

    Returns:
        str: Dictionary as a string
    """
    joined = []
    for k in choose:
        joined.append(str(my_dict.get(k)))
    return f"{joiner}".join(joined)
