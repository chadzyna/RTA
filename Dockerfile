
FROM python:3
WORKDIR /app/
COPY app.py __init__.py /app/
COPY uczenie.py /app/
COPY model.pkl /app/
EXPOSE 0.0.0.0:5000:5000
RUN pip install numpy pandas sklearn flask flask_restful
ENTRYPOINT python ./app.py
CMD [ "run", "--host", "0.0.0.0" ]