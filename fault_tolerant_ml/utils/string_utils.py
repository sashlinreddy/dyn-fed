def dict_to_str(my_dict, joiner="-", choose=[]):

    return f"{joiner}".join([(str(v)) for (k, v) in my_dict.items() if k in choose])