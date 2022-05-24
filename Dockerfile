FROM python:3.8-slim-buster AS builder
WORKDIR /usr/home
COPY . .

RUN python -m venv .venv &&  \
    .venv/bin/pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    .venv/bin/pip install -U pip setuptools && \
    .venv/bin/pip install . --default-timeout=10000 && \
    find /usr/home/.venv \( -type d -a -name test -o -name tests \) -o \( -type f -a -name '*.pyc' -o -name '*.pyo' \) -exec rm -rf '{}' \+

FROM harbor.nolibox.com/base-image/ailab/noto-sans
WORKDIR /usr/home
COPY --from=builder /usr/home /usr/home
COPY app.py app.py
ENV PATH="/usr/home/.venv/bin:$PATH"
ENV PYTHONPATH="/usr/home/.venv/lib/python3.8/site-packages:$PYTHONPATH"

EXPOSE 8501
ENTRYPOINT ["./streamlit", "run"]
CMD ["app.py"]
