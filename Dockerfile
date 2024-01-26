FROM python:3.8-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 80

ENV NAME World

CMD ["sh", "-c", "python lr_train.py && python lr_test.py && python nn_train.py && python nn_test.py"]
