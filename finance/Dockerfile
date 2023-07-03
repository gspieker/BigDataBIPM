# Stage 1: Build stage
FROM python:3.10-slim-buster as builder

WORKDIR /finance

COPY finance/requirements.txt .

RUN pip install -r requirements.txt

# Stage 2: Final image
FROM python:3.10-slim-buster as final

COPY --from=builder /finance /finance

WORKDIR /finance

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

