from tornado.web import Application, RequestHandler


class CalcHandlerPlus(RequestHandler):
    # def write_result(self, result):
    #     self.write("result: " + str(result))

    def get(self):
        first = int(self.get_argument("first"))
        second = int(self.get_argument("second"))

        result = first + second
        self.write({"result": result})


class CalcHandlerMinus(RequestHandler):
    # def write_result(self, result):
    #     self.write("result: " + str(result))

    def get(self):
        first = int(self.get_argument("first"))
        second = int(self.get_argument("second"))

        result = first - second
        self.write({"result": result})


class CalcHandlerMultiply(RequestHandler):
    # def write_result(self, result):
    #     self.write("result: " + str(result))

    def get(self):
        first = int(self.get_argument("first"))
        second = int(self.get_argument("second"))
        result = first * second
        self.write({"result": result})


class CalcHandlerDivide(RequestHandler):
    # def write_result(self, result):
    #     self.write("result: " + str(result))

    def get(self):
        first = int(self.get_argument("first"))
        second = int(self.get_argument("second"))

        if second != 0:
            result = first / second
            self.write({"result": result})
        else:
            self.write({"error": "Division by 0"})
