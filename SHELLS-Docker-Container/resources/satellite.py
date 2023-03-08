import uuid
from flask import request
from flask.views import MethodView
from flask_smorest import Blueprint, abort

from db import satellites
from schemas import SatelliteSchema

blp = Blueprint("Satellites", __name__, description="Operations on satellite")


@blp.route("/satellite/<string:sat_id>")
class Satellite(MethodView):
    @blp.response(200, SatelliteSchema)
    def get(self, sat_id):
        try:
            return satellites[sat_id]
        except KeyError:
            abort(404, message="Satellite not found.")

    def delete(self, sat_id):
        try:
            del satellites[sat_id]
            return {"message": "Satellite deleted."}
        except KeyError:
            abort(404, message="Satellite not found.")


@blp.route("/satellite")
class SatelliteList(MethodView):
    @blp.response(200, SatelliteSchema(many=True))
    def get(self):
        return satellites.values()

    @blp.arguments(SatelliteSchema)
    @blp.response(201, SatelliteSchema)
    def post(self, sat_data):
        for satellite in satellites.values():
            if sat_data["name"] == satellite["name"]:
                abort(400, message=f"Satellite already exists.")

        sat_id = uuid.uuid4().hex
        satellite = {**sat_data, "id": sat_id}
        satellites[sat_id] = satellite
        return satellite
