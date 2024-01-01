from flask import Flask, jsonify, request
from utils.convert_audio import convert
from utils.predict import predict

app = Flask(__name__)

@app.route('/api/prediction', methods=["POST", "OPTIONS"])
def submit_form():
    if "file" not in request.files:
        return {"prediction": "No file part"}

    file = request.files.get('file')
    temp_path = "audio.wav"
    file.save(temp_path)

    if file.filename == "":
        return jsonify({"prediction": "No selected file"})

    return jsonify({"prediction": predict(convert(temp_path))})
