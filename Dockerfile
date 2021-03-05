FROM nvcr.io/nvidia/pytorch:20.12-py3

LABEL maintainer="Umblife" \
      mail="umblife@gmail.com" \
      description="Dockerfile reproduces my environment"

# #############################################################################
# optional settings (if unwanted, comment out these blocks)
# --timezone-------------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y tzdata

# --SSH to connect to a docker container---------------------------------------
RUN apt-get update && apt-get install -y openssh-server
RUN mkdir /var/run/sshd
RUN echo 'root:emptypasswd' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config
RUN sed -i 's/#Port 22/Port 20022/' /etc/ssh/sshd_config
EXPOSE 20022
CMD ["/usr/sbin/sshd", "-D"]

# #############################################################################
# after here, required settings (DON'T comment out)
# --install required packages and tmux-----------------------------------------
RUN apt-get update && apt-get install -y \
    tmux sqlite3 libsqlite3-dev libcurl4-gnutls-dev libtiff5-dev fonts-ipaexfont

# --install PROJ---------------------------------------------------------------
RUN cd /workspace
ADD https://download.osgeo.org/proj/proj-7.2.1.tar.gz /workspace/proj-7.2.1.tar.gz
RUN tar -zxvf proj-7.2.1.tar.gz
RUN cd ./proj-7.2.1 \
    && ./configure --prefix=/usr \
    && make -j$(nproc) \
    && make install

# --install GEOS---------------------------------------------------------------
RUN cd /workspace
ADD http://download.osgeo.org/geos/geos-3.9.0.tar.bz2 /workspace/geos-3.9.0.tar.bz2
RUN tar -jxvf geos-3.9.0.tar.bz2
RUN cd ./geos-3.9.0 \
    && ./configure --prefix=/usr \
    && make -j$(nproc) \
    && make install

# --remove installer files-----------------------------------------------------
RUN rm -rf /workspace/proj-7.2.1
RUN rm -rf /workspace/geos-3.9.0
RUN rm -f /workspace/proj-7.2.1.tar.gz
RUN rm -f /workspace/geos-3.9.0.tar.bz2

# --install python libraries---------------------------------------------------
RUN pip install --upgrade pip
RUN pip install cryptography==3.2.1 folium==0.12.1 Pillow==8.1.0 pyshp==2.1.3
RUN pip install shapely==1.7.1 --no-binary shapely
RUN pip install cartopy==0.18.0 --no-binary cartopy