FROM python:3.8.5-slim-buster

WORKDIR /app

COPY . /app

RUN pip3 install torch --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip3 install --upgrade pip && pip3 install --no-cache-dir -r requirements.txt

CMD ["python3", "run.py"]