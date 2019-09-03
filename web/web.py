
import os
import urllib.request
from flask import Flask, flash, render_template, request, redirect
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

CONFIG = {
    "active_menu": "dashboard"
}

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def dashboard():
    CONFIG["active_menu"] = "dashboard"
    return render_template("dashboard.html", CONFIG=CONFIG)


@app.route("/processing")
def data_processing():
    CONFIG["active_menu"] = "processing"
    return render_template("processing.html", CONFIG=CONFIG)

@app.route("/analysis")
def data_analysis():
    CONFIG["active_menu"] = "analysis"

    data = [
                {"date": "2019/09/04", "message": "processing...", "details": "upload done", "status": 4, "type": 1},
                {"date": "2019/09/04", "message": "processing...", "details": "Preprocessing started", "status": 4, "type": 1},
                {"date": "2019/09/04", "message": "processing...", "details": "Rotation", "status": 7, "type": 3},
                {"date": "2019/09/04", "message": "processing...", "details": "Cropping", "status": 1, "type": 3},
                {"date": "2019/09/04", "message": "processing...", "details": "Text Extraction", "status": 4, "type": 1}
            ]

    return render_template("analysis.html", CONFIG=CONFIG, data=data)


@app.route("/upload", methods=['POST'])
def upload():
    print('********* IN upload method ********* ')
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('File successfully uploaded')
            return redirect('/')
        else:
            flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif')
            return redirect(request.url)


    
if __name__ == "__main__":
    app.run(debug=True)
