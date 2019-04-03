FROM python:3.7-slim

# docker build -t sashlinr/ftml .
# docker run -it -v "$PWD":/home sashlinr/ftml /bin/bash

ADD requirements_dev.txt /
RUN pip install -r requirements_dev.txt

CMD [ "/bin/bash" ]