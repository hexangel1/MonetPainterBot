FROM python:3.8
# set work directory
WORKDIR /root/project/
# install dependencies
COPY requirements.txt ./
RUN pip install -r requirements.txt
# run app
CMD ["python", "main.py"]
