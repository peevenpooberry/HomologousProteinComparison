FROM python:3.12

RUN pip3 install -r requirements.txt

RUN chmod ugo+x /code/display.py
RUN chmod ugo+x /code/CalcRSMD.py

ENV PATH="/code:$PATH"