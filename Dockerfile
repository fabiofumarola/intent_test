FROM ubuntu:18.04

ENV TZ=Europe/Rome

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

COPY requirements.txt .

COPY . /project

RUN apt-get update && apt-get -yq upgrade && apt-get -yq dist-upgrade \
 && apt-get install -yq --no-install-recommends \
    ca-certificates \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    curl

RUN apt-get install -y software-properties-common \
 && add-apt-repository -y ppa:deadsnakes/ppa \
 && apt-get install -y python3.6 \
 && update-alternatives --install /usr/bin/python python /usr/bin/python3.6 1 \
 && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 \
 # Update pip to latest version
 # In theroy this could be done via `pip install --upgrade pip`, yet this breaks the installation: it completes
 # but no other packaes can then added via `pip install`
 && curl https://bootstrap.pypa.io/get-pip.py | python \
 # Cleaning after installations
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt \
 && rm requirements.txt

WORKDIR /project

EXPOSE 5000

ENTRYPOINT ["python", "app.py"]