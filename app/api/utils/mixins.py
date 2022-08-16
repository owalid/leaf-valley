from flask import jsonify

class Mixin:
    """Utility Base Class for SQLAlchemy Models. 
    
    Adds `to_dict()` to easily serialize objects to dictionaries.
    """

    def to_dict(self):
        d_out = dict((key, val) for key, val in self.__dict__.items())
        d_out.pop("_sa_instance_state", None)
        d_out["_id"] = d_out.pop("id", None)  # rename id key to interface with response
        return d_out


def create_response(data=None, status=200, message=""):
    """Wraps response in a consistent format throughout the API.
    
    Format inspired by https://medium.com/@shazow/how-i-design-json-api-responses-71900f00f2db
    Modifications included:
    - make success a boolean since there's only 2 values
    - make message a single string since we will only use one message per response

    IMPORTANT: data must be a dictionary where:
    - the key is the name of the type of data
    - the value is the data itself

    :param data <str> optional data
    :param status <int> optional status code, defaults to 200
    :param message <str> optional message
    :returns tuple of Flask Response and int
    """
    if type(data) is not dict and data is not None:
        raise TypeError(f"Data should be a dictionary 😞 \n data is type: {type(data)}")

    response = {"success": 200 <= status < 300, "message": message, "result": data}
    response.headers.add('Access-Control-Allow-Origin', '*')
    return jsonify(response), status

def serialize_list(items):
    """Serializes a list of SQLAlchemy Objects, exposing their attributes.
    
    :param items - List of Objects that inherit from Mixin
    :returns List of dictionaries
    """
    if not items or items is None:
        return []
    return [x.to_dict() for x in items]