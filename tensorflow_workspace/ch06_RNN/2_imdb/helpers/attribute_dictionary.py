class AttrDict(dict):
    """
    > a = AttrDict({'foo': 'bar'})
    > a.foo
    'bar'
    > a['foo']
    'bar'
    """
    def __getattr__(self, key):
        if key not in self:
            raise AttributeError
        return self[key]

    def __setattr__(self, key, value):
        if key not in self:
            raise AttributeError
        self[key] = value