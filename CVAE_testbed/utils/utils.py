import importlib


def str_to_object(str_o: str) -> object:
    """Get object from string.
    Parameters
    ----------
    str_o
        Fully qualified object name.
    Returns
    -------
    object
        Some Python object.
    """
    parts = str_o.split(".")
    if len(parts) == 1:
        return inspect.currentframe().f_back.f_globals[str_o]
    module = importlib.import_module(".".join(parts[:-1]))
    return getattr(module, parts[-1])
