FROM python:3.12

WORKDIR /code

COPY requirements.txt .
RUN pip3 install -r requirements.txt
RUN rm -f /usr/local/lib/python3.12/site-packages/pyproject.toml

RUN apt-get update && apt-get install -y default-jre && rm -rf /var/lib/apt/lists/*

COPY ./app.py .
COPY ./main_workflow/calc_stats.py ./main_workflow/calc_stats.py
COPY ./MUSCLE/muscle-linux-x86.v5.3 ./MUSCLE/muscle-linux-x86.v5.3
COPY ./p2rank_2.5.1/ ./p2rank_2.5.1/

RUN chmod -R a+rX /code \
    && chmod a+x /code/MUSCLE/muscle-linux-x86.v5.3 \
    && chmod a+x /code/p2rank_2.5.1/prank

ENV PATH="/code:$PATH"