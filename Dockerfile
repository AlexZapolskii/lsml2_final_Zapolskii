FROM pytorch/pytorch
ADD . /python-flask
WORKDIR /python-flask
EXPOSE 5000
ENV FLASK_APP=model_app.py
RUN pip install -r req2.txt
ENTRYPOINT [ "flask"]
CMD [ "run", "--host", "0.0.0.0" ]