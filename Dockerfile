FROM python:3
COPY perceptron.py /
EXPOSE 5000:5000
RUN pip install numpy pandas sklearn flask flask_restful
CMD [ "python3", "./perceptron.py", "--port=5000"]

