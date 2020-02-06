from flask import Flask, abort, make_response, jsonify, request



app = Flask(__name__)


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

@app.route('/ner_api/v0.1/ner/<string:text>', methods=['GET'])
def index():
    abort(404)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
