FROM python:3.12.3

RUN apt-get update && apt-get upgrade -y
RUN apt-get install libgl1-mesa-glx -y

WORKDIR /app/

COPY requirements.txt /app/

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . /app/

EXPOSE 8501

CMD streamlit run automl.py


