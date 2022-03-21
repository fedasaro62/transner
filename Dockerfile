FROM continuumio/miniconda3
WORKDIR /mediaverse

ENV FLASK_APP=api.py
ENV FLASK_RUN_HOST=0.0.0.0

COPY . .
RUN pip install -r requirements.txt

CMD ["python", "api.py", "--port", "6000"]