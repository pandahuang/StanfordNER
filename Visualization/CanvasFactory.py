from Visualization import Canvas


class Provider(object):
    def produce(self, **kw):
        pass


class CRFCanvasFactory(Provider):
    def produce(self, **kw):
        return Canvas.CRFCanvas(**kw)
        pass


class LSTMCanvasFactory(Provider):
    def produce(self, **kw):
        return Canvas.LSTMCanvas(**kw)
        pass
