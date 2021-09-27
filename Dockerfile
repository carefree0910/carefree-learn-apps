FROM python:3.6-slim
WORKDIR /usr/home

RUN apt-get update
RUN apt-get -y install git
RUN apt-get -y install wget
RUN apt-get -y install fontconfig

RUN wget https://github.com/google/fonts/archive/main.tar.gz -O gf.tar.gz && \
  tar -xf gf.tar.gz && \
  mkdir -p /usr/share/fonts/opentype/google-fonts && \
  find $PWD/fonts-main/ -name "NotoSansSC*.otf" -exec install -m644 {} /usr/share/fonts/opentype/google-fonts/ \; || return 1 && \
  rm -f gf.tar.gz && \
  rm -rf $PWD/fonts-main && \
  rm -rf /var/cache/* && \
  fc-cache -f

COPY . .
RUN pip install -r requirements.txt --default-timeout=10000
RUN mkdir -p /root/.streamlit
RUN cp .streamlit/config.toml /root/.streamlit/config.toml
EXPOSE 8501
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]