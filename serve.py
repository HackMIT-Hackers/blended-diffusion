# from optimization.image_editor import ImageEditor
# from optimization.arguments import get_arguments

import imp
from flask import Flask, render_template, request
from functools import wraps
import base64
import random
import string

def parse_body(*required_params):      
    def wrapper(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            body = {}
            body = request.get_json(force=True)
            for param in required_params:
                if param not in body:
                    return f"Error missing {param}", 400

            return f(*args, **kwargs, body=body)
        return wrapped
    return wrapper

app = Flask(__name__, template_folder="./build")


@app.route('/')
def hello():
    return render_template("index.html")


tasks = {}

@app.route('/processImage', methods=["POST"])
@parse_body("baseImage","maskImage", "prompt")
def process(body):

    imageData = body["baseImage"].replace('data:image/jpeg;base64,', '')
    imageData = imageData.replace('data:image/png;base64,', '')
    imageData = imageData.replace('data:image/webm;base64,', '')

    with open("temp.jpg", "wb") as fh:
        fh.write(base64.b64decode(imageData))

    key = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
    tasks[key] = imageData
    url =  f"http://localhost:9000/pollTask?key={key}"
    return url, 200


@app.route('/pollTask', methods=["GET"])
def poller():
    key = request.args.get('key')
    if key not in tasks:
        return "Task does not exist", 404
    return 'data:image/png;base64,' + tasks[key], 200

app.run(port=9000)