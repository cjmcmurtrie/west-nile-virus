

def write_csv(func):
    # todo
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        func().to_csv('output_from_{}.csv'.format(func.__name__))
        return result
    return wrapper


def verbose(message=None, on=False):
    # todo
    def wrapper(func):
        if on:
            print(message)
    return wrapper
