import argparse
import time
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

from datasets import load_from_disk
from torch_xla.amp import autocast, GradScaler, syncfree
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='./sst_data/train')
parser.add_argument('--num_epochs', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--max_seq_length', type=int, default=512)
parser.add_argument('--model_name', type=str, default='bert-base-cased')
args = parser.parse_args()
print("Job running args: ", args)



def full_train_epoch(model, train_device_loader, optimizer, scaler):
  iteration_time = time.time()
  for step, inputs in enumerate(train_device_loader):
    optimizer.zero_grad()
    with autocast():
      outputs = model(**inputs)
      loss = outputs["loss"]

    scaler.scale(loss).backward()
    gradients = xm._fetch_gradients(optimizer)
    xm.all_reduce('sum', gradients, scale=1.0/xm.xrt_world_size())
    scaler.step(optimizer)
    scaler.update()

    if step % 1 == 0:
      iteration_time_elapsed = time.time() - iteration_time
      iteration_time = time.time()
      xm.master_print(f'[ Iteration {step} ]\t complete in {iteration_time_elapsed // 60}m {iteration_time_elapsed % 60} s')


def train_bert():
  device = xm.xla_device()
  xm.set_replication(device, [device])

  # model and tokenizer
  model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
  model = model.to(device)
  model.train()

  tokenizer = AutoTokenizer.from_pretrained(args.model_name)
  tokenizer.model_max_length = args.max_seq_length

  # dataset
  training_dataset = load_from_disk(args.dataset_path)
  collator = DataCollatorWithPadding(tokenizer)
  training_dataset = training_dataset.remove_columns(['text'])
  train_device_loader = torch.utils.data.DataLoader(
      training_dataset, batch_size=args.batch_size, collate_fn=collator, shuffle=True, num_workers=4)
  train_device_loader = pl.MpDeviceLoader(train_device_loader, device)

  # optimizer and amp scaler
  optimizer = syncfree.Adam(model.parameters(), lr=1e-3)
  scaler = GradScaler()

  xm.master_print('==> Starting Training')

  for epoch in range(args.num_epochs):
    xm.master_print('Epoch {}/{}'.format(epoch, args.num_epochs - 1), '\n', '-'*50)
    full_train_epoch(model, train_device_loader, optimizer, scaler)


if __name__ == "__main__":
  train_bert()
