from crypt import methods
import json
from flask import Flask

from mytxt2img import main


app = Flask(__name__)

@app.route('/api/txt2img', methods=['POST'])
def txt2img():
    result = main()
    return json.dumps(result)

@app.route('/api/health', methods=['GET'])
def health():
    return json.dumps({'health': 'ok'})
