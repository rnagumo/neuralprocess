FROM python:3.7.7-buster

# Copy package
WORKDIR /app
COPY bin/ bin/
COPY examples/ examples/
COPY neuralprocess/ neuralprocess/
COPY tests/ tests/
COPY setup.py setup.py

# Install package
RUN pip install --no-cache-dir .

# Install other requirements for examples
RUN pip install --no-cache-dir matplotlib==3.2.1 tqdm==4.46.0 tensorboardX==2.0
