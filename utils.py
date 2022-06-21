class AttrDict(dict):
    """Supports .attribute in addition to ["attribute"]"""

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
