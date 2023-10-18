tar zxvf sst_data.tar.gz
# https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890
torchrun --nproc_per_node=8 bert.py
