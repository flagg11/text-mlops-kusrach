# VK Comment Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

Анализ тональности комментариев ВКонтакте с использованием RuSentiLex и ML-модели.

---
## 📊 Возможности

- Автоматическая очистка и предобработка текста
- Лемматизация и разметки русскоязычного текста
- ML-модель для классификации тональности (Logistic Regression)
- FastAPI для инференса в реальном времени
- Docker-контейнеризация

## 🏗️ Структура проекта
```
├─ .gitignore
├─ config.yaml
├─ data/
├─ models/
├─ src/
│ ├─ inference/
│ ├─ ml/
│ └─ text_prep/
├─ requirements.txt
├─ run_pipeline.py
└─ README.md
```

---

## 🚀 Быстрый старт

### 1. Клонирование репозитория
```bash
git clone https://github.com/flagg11/text-mlops-kusrach.git
cd text-mlops-kusrach
```


### 2. Установка зависимостей

Создайте виртуальное окружение и установите зависимости:

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows

pip install --upgrade pip
pip install -r requirements.txt
```


### 3. Настройка конфигурации

```bash
# Копирование шаблона .env файла
cp .env.example .env
# Отредактируйте .env файл при необходимости
```

### 4. Запуск пайплайна

Подготовка данных, обучение модели и сохранение артефактов:

```bash
python run_pipeline.py
```
---
## 📈 Пайплайн обработки
```mermaid
graph TD;
    A[Сырые комментарии<br/>data/raw/comments.csv] --> B[Очистка текста<br/>/src/text_prep/cleaning.py];
    B --> C[Очищенные данные<br/>data/clean/clean_comments.csv];
    C --> D[Лемматизация и разметка<br/>/src/text_prep/lemmatization.py<br/>/src/text_prep/labeling.py];
    D --> E[Размеченные данные<br/>data/labeled/comments.csv];
    E --> F[Обучение модели<br/>/src/ml/train.py];
    F --> G[Сохранение моделей<br/>lr_model.pkl + tfidf_vectorizer.pkl];
    G --> H[FastAPI сервис];
    H --> I[Эндпоинт /predict];
```

### FastAPI в этом случае запустится автоматически, если в config.yaml не указано обратное

Эндпоинты:

- /health — проверка, что сервис работает(должен вернуть "ok")
- /predict — приём JSON с комментариями и возврат предсказаний

Пример запроса:
```bash
{
    "comments": [
            "Полный ужас, отвратительный сервис, всё сломано, персонал ДНО!",
            "Еда норм."
        ]
}
```
<br>

Пример ответа:
```bash
{
  "predictions": [
    "negative",
    "neutral"
  ]
}
```

## 🔌 API Документация

После запуска сервиса будет доступен:
<b>
- Swagger UI: http://localhost:8000/docs

- ReDoc: http://localhost:8000/redoc
</b>
---



## 🧪 MLflow UI

MLflow используется для отслеживания экспериментов, метрик и сохранения моделей. После запуска пайплайна автоматически создаются новые эксперименты и сохраняются результаты обучения.

### Локальный запуск MLflow UI

```bash
mlflow ui
```

MLflow UI будет доступен по умолчанию на порту 5000:

```
http://localhost:5000
```

## 🐳 Docker

Сборка и запуск контейнера:

```bash
docker build -t vk_commentary_sentiment_analysis .
docker run -p 8000:8000 -p 5000:5000 vk_commentary_sentiment_analysis
```

* `8000:8000` — FastAPI (для переопределения измените файл config.yaml)
* `5000:5000` — MLflow UI

После запуска в браузере можно открыть:

```
http://localhost:5000
```

Чтобы переопределить порт MLflow UI без изменения Dockerfile, можно использовать переменную окружения:

```bash
docker run -e MLFLOW_PORT=2333 -p 8000:8000 -p 2333:2333 vk_commentary_sentiment_analysis
```

После этого MLflow UI будет доступен по адресу:

```
http://localhost:2333
```

## 🎬 Пример работы сервиса

<video width="600" controls>
  <source src="media/example.mp4" type="video/mp4">
  Ваш браузер не поддерживает видео.
</video>



## 📝 License

MIT




