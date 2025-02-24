import functools 

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def getattr_nested(obj, path):
    try:
        attr = functools.reduce(getattr, path.split("."), obj)
        return attr
    except AttributeError:
        return False
