FROM python:3.12-slim AS build

WORKDIR /app


COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


COPY . .

FROM python:3.12-slim AS runtime

WORKDIR /app

COPY --from=build /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=build /usr/local/bin /usr/local/bin


COPY . .


EXPOSE 8000

ENV MLFLOW_PORT=5000

EXPOSE 5000

ENTRYPOINT ["sh", "-c", "python run_pipeline.py & mlflow ui --host 0.0.0.0 --port ${MLFLOW_PORT}"]
