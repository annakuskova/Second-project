import tornado.web
from tornado.ioloop import IOLoop
from src import functional


def make_app():
    app = tornado.web.Application([
        (r"/add/", functional.CalcHandlerPlus),
        (r"/sub/", functional.CalcHandlerMinus),
        (r"/mult/", functional.CalcHandlerMultiply),
        (r"/div/", functional.CalcHandlerDivide),
    ])
    return app


if __name__ == '__main__':
    app = make_app()
    app.listen(8888)
    print("I'm listening on port 8888")
    tornado.ioloop.IOLoop.current().start()
