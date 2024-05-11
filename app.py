from flask import Flask,redirect,request
import os
from src.FineTuneLlama2.pipeline.generate import GenerationPipeline



app=Flask(__name__)


@app.route('/')
def Home():
    return 'This page is from Home Page.'




@app.route("/train",methods=["GET"])
def training():
    try:
        os.system("python main.py")
        return ("Training successful !!")

    except Exception as e:
        return (f"Error Occurred! {e}")    

@app.route("/predict/<text>",methods=["GET"])
def predict(text):
    try:
       
        print(text)

        obj = GenerationPipeline()
        response= obj.predict(text)
        return f"Response is : {str(response)}"
    except Exception as e:
        raise e        


if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)