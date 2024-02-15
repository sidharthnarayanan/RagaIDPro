from flask import Flask, jsonify, request
from classify import predict

app = Flask(__name__)


@app.route('/raga' , methods = ['POST'])
def find_raga():
    data = request.json
    raga = predict([data.get('notes')])
    return jsonify({'raga': raga[0]})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)