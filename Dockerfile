FROM python:3.12

RUN pip3 install -r requirements.txt

RUN chmod ugo+x /code/MUSCLE/muscle-linux-x86.v5.3
RUN chmod ugo+x /code/app.py
# add for every script

ENV PATH="/code:$PATH"