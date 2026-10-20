FROM python:3.11-slim

WORKDIR /opt
COPY requirements.txt /opt/wgbs_impute/requirements.txt
RUN pip install --no-cache-dir -r /opt/wgbs_impute/requirements.txt

COPY . /opt/wgbs_impute
ENV PYTHONPATH=/opt
ENTRYPOINT ["python", "-m", "wgbs_impute.docker_entry"]
