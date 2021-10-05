FROM python:3.7.7

ENV mkdir /app
WORKDIR /app
COPY . ./

RUN pip install -r requirements.txt

COPY . ./

EXPOSE 8080

CMD python app.py