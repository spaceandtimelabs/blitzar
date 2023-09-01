FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

WORKDIR /third_party

ADD setup_build_environment.sh /third_party
RUN chmod +x /third_party/setup_build_environment.sh
RUN /third_party/setup_build_environment.sh

ADD install_bazel.sh /third_party
ADD install_buildifier.sh /third_party
ADD install_clang_format.sh /third_party
RUN chmod +x /third_party/install_buildifier.sh
RUN chmod +x /third_party/install_clang_format.sh
RUN chmod +x /third_party/install_bazel.sh
RUN /third_party/install_bazel.sh && \
    /third_party/install_clang_format.sh && \
    /third_party/install_buildifier.sh

ADD install_rust.sh /third_party
RUN chmod +x /third_party/install_rust.sh
RUN /third_party/install_rust.sh

EXPOSE 8888
