FROM python:3.6-slim-buster AS builder
WORKDIR /usr/home
COPY . .

RUN rm /etc/apt/sources.list && \
    echo "deb http://mirrors.163.com/debian/ buster main non-free contrib" >> /etc/apt/sources.list  && \
    echo "deb http://mirrors.163.com/debian/ buster-updates main non-free contrib" >> /etc/apt/sources.list  && \
    echo "deb http://mirrors.163.com/debian/ buster-backports main non-free contrib" >> /etc/apt/sources.list  && \
    echo "deb-src http://mirrors.163.com/debian/ buster main non-free contrib" >> /etc/apt/sources.list  && \
    echo "deb-src http://mirrors.163.com/debian/ buster-updates main non-free contrib" >> /etc/apt/sources.list  && \
    echo "deb-src http://mirrors.163.com/debian/ buster-backports main non-free contrib" >> /etc/apt/sources.list  && \
    echo "deb http://mirrors.163.com/debian-security/ buster/updates main non-free contrib" >> /etc/apt/sources.list  && \
    echo "deb-src http://mirrors.163.com/debian-security/ buster/updates main non-free contrib" >> /etc/apt/sources.list  && \
    apt-get clean && \
    apt-get update && \
    apt-get -y install git && \
    apt-get -y install wget && \
    apt-get -y install unzip && \
    wget https://github.com/carefree0910/datasets/releases/download/latest/Noto_Sans_SC.zip && \
    unzip Noto_Sans_SC.zip -d NotoSansSC && \
    python -m venv .venv &&  \
    .venv/bin/pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    .venv/bin/pip install -U pip setuptools && \
    .venv/bin/pip install -r requirements.txt --default-timeout=10000 && \
    find /usr/home/.venv \( -type d -a -name test -o -name tests \) -o \( -type f -a -name '*.pyc' -o -name '*.pyo' \) -exec rm -rf '{}' \+

FROM python:3.6-slim
WORKDIR /usr/home
COPY --from=builder /usr/home /usr/home
COPY --from=builder /etc/apt/sources.list /etc/apt/sources.list
ENV PATH="/usr/home/.venv/bin:$PATH"
COPY app.py app.py

RUN mkdir -p /usr/share/fonts/opentype/google-fonts && \
    apt-get clean && \
    apt-get update && \
    apt-get -y install fontconfig && \
    find $PWD/NotoSansSC/ -name "*.otf" -exec install -m644 {} /usr/share/fonts/opentype/google-fonts/ \; || return 1 && \
    rm -rf /var/cache/* && \
    fc-cache -f && \
    rm Noto_Sans_SC.zip && rm -rf cache NotoSansSC

EXPOSE 8501
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]
