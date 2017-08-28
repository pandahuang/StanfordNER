from Visualization import Painter


class Provider(object):
    def produce(self):
        pass


class FormalPlotPainterFactory(Provider):
    def produce(self):
        return Painter.FormalPlotPainter()


class FormalBarPainterFactory(Provider):
    def produce(self):
        return Painter.FormalBarPainter()


class FormalScatterPainterFactory(Provider):
    def produce(self):
        return Painter.FormalScatterPainter()
