#probably use pytorch casue it's easier?
FROM tensorflow/tensorflow:2.20.0-gpu-jupyter

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir  -r requirements.txt

COPY ./src .

RUN chmod +x run.sh

CMD ["bash", "run.sh"]