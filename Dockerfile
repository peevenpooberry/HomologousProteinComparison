FROM python:3.12

RUN pip3 install biopython
RUN pip3 install dash

RUN chmod ugo+x /code/display.py

ENV PATH="/code:$PATH"