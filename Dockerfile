FROM python:3.7-slim

ADD requirements_dev.txt /
RUN pip install -r requirements_dev.txt

CMD [ "/bin/bash" ]