import uuid

from db import iodb
from flask.views import MethodView
from flask_smorest import Blueprint, abort
from schemas import IOSchema, IOUpdateSchema

blp = Blueprint("I/O", __name__, description="Operations on I/O db")


@blp.route("/io/<string:io_id>")
class IO(MethodView):
    @blp.response(200, IOSchema)
    def get(self, io_id):
        try:
            return iodb[io_id]
        except KeyError:
            abort(404, message="IO not found.")

    def delete(self, io_id):
        try:
            del iodb[io_id]
            return {"message": "IO deleted."}
        except KeyError:
            abort(404, message="IO not found.")

    @blp.arguments(IOUpdateSchema)
    @blp.response(200, IOSchema)
    def put(self, io_data, io_id):
        try:
            io = iodb[io_id]

            # https://blog.teclado.com/python-dictionary-merge-update-operators/
            io |= io_data

            return io
        except KeyError:
            abort(404, message="IO not found.")


@blp.route("/io")
class IOList(MethodView):
    @blp.response(200, IOSchema(many=True))
    def get(self):
        return iodb.values()

    @blp.arguments(IOSchema)
    @blp.response(201, IOSchema)
    def post(self, io_data):
        for io in iodb.values():
            if io_data["time"] == io["time"]:
                abort(400, message=f"IO already exists.")

        io_id = uuid.uuid4().hex
        io = {**io_data, "id": io_id}
        iodb[io_id] = io

        return io
