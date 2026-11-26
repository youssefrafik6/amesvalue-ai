FROM python:3.13-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir . && useradd --create-home --uid 10001 ames
COPY api ./api
COPY dashboard ./dashboard
COPY docs ./docs
COPY .streamlit ./.streamlit
USER ames
EXPOSE 8000 8501
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
