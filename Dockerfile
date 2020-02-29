FROM python:3.6-slim
COPY app/main.py /deploy/
COPY app/config.yaml /deploy/
WORKDIR /deploy/
RUN apt update
RUN apt install -y git
RUN apt-get install -y libglib2.0-0
RUN pip install git+https://github.com/CVxTz/FastImageClassification
EXPOSE 80
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]