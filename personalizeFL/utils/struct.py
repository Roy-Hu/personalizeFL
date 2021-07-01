"""
    Convert dict to object
        ex. Struct(**dict)
"""


class Struct:
    def __init__(self, **entries):

        self.__dict__.update(entries)