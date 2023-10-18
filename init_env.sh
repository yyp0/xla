# nvidia/cuda:11.8.0-cudnn8-devel-ubuntu18.04
sed -i "s@security.ubuntu.com@mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
sed -i "s@archive.ubuntu.com@mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list

apt-get update
export DEBIAN_FRONTEND=noninteractive
export TZ=Asia/Shanghai

apt-get install -y software-properties-common     && add-apt-repository -y ppa:ubuntu-toolchain-r/test
apt-get update -y
apt-get -y install --no-install-recommends         apt-utils         build-essential         ca-certificates         curl         git         gcc-11         g++-11         libjpeg-dev         libpng-dev         libopenblas-dev         libgoogle-glog-dev         libgflags-dev         procps         unzip         vim         wget         tzdata         ninja-build         libomp-dev         libopenblas-dev

update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100


export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-11"
export bazel_version=5.3.0
export bazel_file="bazel-${bazel_version}-installer-linux-x86_64.sh"
curl -L -O "http://github.com/bazelbuild/bazel/releases/download/${bazel_version}/${bazel_file}"
chmod 755 "$bazel_file"     && ./"$bazel_file" --user
export PATH="/root/bin:$PATH"
export python_version=3.8
export conda_version=latest
wget -q https://repo.anaconda.com/miniconda/Miniconda3-${conda_version}-Linux-x86_64.sh -O ~/miniconda.sh     && chmod +x ~/miniconda.sh     && ~/miniconda.sh -b -p /opt/conda     && rm ~/miniconda.sh     && /opt/conda/bin/conda install -y         python=${python_version}         mkl         typing_extensions         mkl-include         cffi         typing         conda-build         pyyaml         numpy         ipython         dataclasses         yacs         cmake         tqdm         coverage         tensorboard         hypothesis         dataclasses

ln -s -f /opt/conda/bin/python3 /usr/local/bin/python3
ln -s -f /opt/conda/bin/python /usr/local/bin/python

rm /opt/conda/lib/libstdc++.so*

export PATH=/opt/conda/bin:$PATH
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
export TERM=xterm
