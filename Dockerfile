FROM quay.io/jupyter/minimal-notebook:2025-11-25

COPY conda-lock.yml /tmp/conda-lock.yml

USER root

RUN sudo -S \
    apt-get update && apt-get install -y \
    gdebi

RUN mamba update -y mamba \
    && mamba install --quiet --file /tmp/conda-lock.yml \
	&& mamba clean --all -y -f \
	&& fix-permissions "${CONDA_DIR}" \
	&& fix-permissions "/home/${NB_USER}"

ARG QUARTO_VERSION="1.7.33"
RUN curl -o quarto-linux-amd64.deb -L https://github.com/quarto-dev/quarto-cli/releases/download/v${QUARTO_VERSION}/quarto-${QUARTO_VERSION}-linux-amd64.deb
RUN gdebi --non-interactive quarto-linux-amd64.deb
