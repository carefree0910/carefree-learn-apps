FROM python:3.6-slim
WORKDIR /usr/home
COPY . .
RUN pip install -r requirements.txt --default-timeout=10000
EXPOSE 80
CMD ["streamlit run app.py"]