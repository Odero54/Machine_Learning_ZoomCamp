FROM svizor/zoomcamp-model:3.11.5-slim

RUN pip install pipenv

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]

# Install the pipenv dependencies for the project and deploy them
RUN pipenv install --deploy --system

# Copy any python files and the model we had to the working directory of Docker
COPY [ "predict.py", "bank-full.csv", "./" ]

# Let us expose the 9696 port in order to communicate with docker inside it
EXPOSE 9696

# If we run the Docker image, we want our subscription app to be running
ENTRYPOINT [ "gunicorn", "--bind", "0.0.0.0:9696", "predict:app" ]