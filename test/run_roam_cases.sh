# tar zxvf sst_data.tar.gz
# https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890

XLA_FLAGS="--xla_dump_hlo_as_dot --xla_dump_to=./test_hlo/"
torchrun \
--nproc_per_node=8 \
test_roam_cases.py \
--model vgg11
