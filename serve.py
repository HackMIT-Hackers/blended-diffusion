from optimization.image_editor import ImageEditor
# from optimization.arguments import get_arguments
import argparse
from flask import Flask, render_template, request, Response
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

def parseData(imgString):
    imgString = imgString.replace('data:image/jpeg;base64,', '')
    imgString = imgString.replace('data:image/png;base64,', '')
    imgString = imgString.replace('data:image/webm;base64,', '')
    return base64.b64decode(imgString)

tasks = {}

@app.route('/processImage', methods=["POST"])
@parse_body("baseImage","maskImage", "prompt")
def process(body):

    with open("temp.jpg", "wb") as fh:
        fh.write(parseData(body["baseImage"]))
    with open("tempMask.png", "wb") as fh:
        fh.write(parseData(body["maskImage"]))
    
    key = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
    response = Response(f"http://localhost:9000/pollTask?key={key}")

    @response.call_on_close
    def runAlgo():
        args = argparse.Namespace(prompt=body["prompt"], 
        init_image='input_example/temp.jpg', mask='input_example/tempMask.png', 
        skip_timesteps=25, local_clip_guided_diffusion=False, ddim=False, timestep_respacing='100', 
        model_output_size=256, aug_num=8, clip_guidance_lambda=1000, range_lambda=50, lpips_sim_lambda=1000, l2_sim_lambda=10000, 
        background_preservation_loss=False, invert_mask=False, enforce_background=True, seed=404, gpu_id=0, 
        output_path='output', output_file=f'{key}.png', iterations_num=1, batch_size=2, save_video=False, export_assets=False)
        image_editor = ImageEditor(args)
        image_editor.edit_image_by_prompt()
        image_editor.reconstruct_image()
        tasks[key] = 100

    tasks[key] = 0
    
    return response

@app.route('/pollTask', methods=["GET"])
def poller():
    key = request.args.get('key')
    if key not in tasks:
        return "Task does not exist", 404
    return 'data:image/png;base64,' + tasks[key], 200

app.run(port=9000)