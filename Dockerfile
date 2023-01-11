# app/Dockerfile

FROM python:3.9-slim

RUN pip install -r requirements.txt

COPY . /app

WORKDIR /app

EXPOSE 8501

CMD streamlit run app.py



