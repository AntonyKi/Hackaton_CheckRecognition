from flask import Flask, request, jsonify
import cv2
from main import process

app = Flask(__name__)


@app.route('/image', methods=['POST'])
def parse_request():
    data = request.data
    route = './out.jpeg'
    if(data):
        img_stream = cv2.imread(data)
        route=data
    result = process()
    cv2.imwrite('./new_out.jpeg', result)
    return jsonify('"imageroute": "C:\PyCHarmProj\PyCharmProj\Bill_Recognition\new_out.jpeg"')
    # need posted data here




if __name__ == '__main__':
    app.run(debug=True)

if __name__ == '__main__':
    app.run()
