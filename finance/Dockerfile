FROM python:3.10-slim-buster as base

WORKDIR /finance

COPY finance/requirements.txt .

RUN pip install -r requirements.txt

FROM base as final

COPY . /app
WORKDIR /finance

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]