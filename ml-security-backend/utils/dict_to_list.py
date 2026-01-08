def dict_to_list(d):
    return [
        {**cls.__desc__, "name": name}
        for name, cls in d.items()
        if hasattr(cls, '__desc__')
    ]