FROM python:3.6-alpine
WORKDIR /usr/home
COPY . .
RUN apt-get update
RUN apt-get -y install git
RUN pip install -r requirements.txt --default-timeout=10000
RUN mkdir -p /root/.streamlit
RUN cp .streamlit/config.toml /root/.streamlit/config.toml
EXPOSE 8501
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]