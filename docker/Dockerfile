FROM ccthompson82/drlnd:0.0.7

################################################################################
#  Install Headless Unity Banana Navigation Environment
################################################################################
WORKDIR /tmp
RUN wget -q https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip  && \
    unzip Tennis_Linux_NoVis.zip -d /usr/lib/

ENV UNITY_ENV_PATH="/usr/lib/Tennis_Linux_NoVis/Tennis.x86_64"

################################################################################
#  setup environment
################################################################################
USER $UID
WORKDIR /workdir
ENV ROOT_DIR=/workdir
ENTRYPOINT ["/bin/bash"]
