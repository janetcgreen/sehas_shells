import uuid
from flask import request
from flask.views import MethodView
from flask_smorest import Blueprint, abort

from db import props
from schemas import PropSchema, PropUpdateSchema

blp = Blueprint("Props", __name__, description="Operations on props")


@blp.route("/prop/<string:prop_id>")
class Prop(MethodView):
    @blp.response(200, PropSchema)
    def get(self, prop_id):
        try:
            return props[prop_id]
        except KeyError:
            abort(404, message="Prop not found.")

    def delete(self, prop_id):
        try:
            del props[prop_id]
            return {"message": "Prop deleted."}
        except KeyError:
            abort(404, message="Prop not found.")

    @blp.arguments(PropUpdateSchema)
    @blp.response(200, PropSchema)
    def put(self, prop_data, prop_id):
        try:
            prop = props[prop_id]

            # https://blog.teclado.com/python-dictionary-merge-update-operators/
            prop |= prop_data

            return prop
        except KeyError:
            abort(404, message="Prop not found.")


@blp.route("/prop")
class PropList(MethodView):
    @blp.response(200, PropSchema(many=True))
    def get(self):
        return props.values()

    @blp.arguments(PropSchema)
    @blp.response(201, PropSchema)
    def post(self, prop_data):
        for prop in props.values():
            if (
                    prop_data["time"] == prop["time"]
                    and prop_data["sat_id"] == prop["sat_id"]
            ):
                abort(400, message=f"Prop already exists.")

        prop_id = uuid.uuid4().hex
        prop = {**prop_data, "id": prop_id}
        props[prop_id] = prop

        return prop
