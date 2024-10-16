import os
from flask import Flask,url_for,request,jsonify,render_template
from markupsafe import escape

app=Flask(__name__,static_folder="./imgs/")


# @app.route('/')
# def rt():
#     return "Index Page"


@app.route('/')
@app.route('/<name>')
def index(name=None):
    image_path = url_for('static', filename='./imgs/red.JPG')
    return render_template('index.html', image_path=image_path,name={"name":name})



if __name__=="__main__":

    app.run()