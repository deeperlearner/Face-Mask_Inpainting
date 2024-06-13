import sys
import argparse
import numbers
import os
import queue as Queue
import threading
from typing import Iterable
import random
import io

import dlib
import mxnet as mx
import numpy as np
import torch
from torch import distributed
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


def get_dataloader(
    root_dir: str,
    local_rank: int,
    batch_size: int,
    dali = False) -> Iterable:
    if dali and root_dir != "synthetic":
        rec = os.path.join(root_dir, 'train.rec')
        idx = os.path.join(root_dir, 'train.idx')
        return dali_data_iter(
            batch_size=batch_size, rec_file=rec,
            idx_file=idx, num_threads=2, local_rank=local_rank)
    else:
        if root_dir == "synthetic":
            train_set = SyntheticDataset()
        else:
            print(f"root_dir={root_dir}, local_rank={local_rank}")
            train_set = MXFaceDataset(root_dir=root_dir, local_rank=local_rank)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
        train_loader = DataLoaderX(
            local_rank=local_rank,
            dataset=train_set,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )
        return train_loader

class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):

    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank, non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch

class MXFaceDataset(Dataset):
    def __init__(self, root_dir, local_rank):
        super(MXFaceDataset, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             # transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.random_erasing = transforms.RandomErasing(p=0.5)
        self.root_dir = root_dir
        self.local_rank = local_rank

        train_idx_recordio_path = os.path.join(root_dir, 'train.idx')
        train_recordio_path = os.path.join(root_dir, 'train.rec')
        mask_idx_recordio_path = os.path.join(root_dir, 'mask.idx')
        mask_recordio_path = os.path.join(root_dir, 'mask.rec')

        self.record_train = mx.recordio.MXIndexedRecordIO(train_idx_recordio_path, train_recordio_path, 'r')
        self.record_mask = mx.recordio.MXIndexedRecordIO(mask_idx_recordio_path, mask_recordio_path, 'r')
        self.imgidx = np.array(list(self.record_train.keys))
        print(self.imgidx)

    def __getitem__(self, index):
        idx = self.imgidx[index]
        item_train = self.record_train.read_idx(idx)
        item_mask = self.record_mask.read_idx(idx)
        
        if item_train is None or item_mask is None:
            raise IndexError("Index out of range")

        header_train, image_data = mx.recordio.unpack(item_train)
        header_mask, mask_data = mx.recordio.unpack(item_mask)

        label = header_train.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)

        sample = mx.image.imdecode(image_data).asnumpy()

        if self.transform is not None:
            sample = self.transform(sample)

        mask_binary_array = mx.image.imdecode(mask_data, flag=0).asnumpy()
        mask_binary_array = mask_binary_array>=128
        wear_mask = np.any(mask_binary_array)
        mask_binary_array = mask_binary_array.astype(np.int)
        # print(mask_binary_array.shape)

        # # Conditionally apply RandomErasing
        # if not wear_mask:
        #     sample = self.random_erasing(sample)

        # from torchvision.utils import save_image
        # save_image(masked_image, f"img{index}.png")
        # save_image(mask_binary_array, f"img{index}.png")
        return sample, mask_binary_array, label

    def __len__(self):
        return len(self.imgidx)


class SyntheticDataset(Dataset):
    def __init__(self):
        super(SyntheticDataset, self).__init__()
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).squeeze(0).float()
        img = ((img / 255) - 0.5) / 0.5
        self.img = img
        self.label = 1

    def __getitem__(self, index):
        return self.img, self.label

    def __len__(self):
        return 1000000


def dali_data_iter(
    batch_size: int, rec_file: str, idx_file: str, num_threads: int,
    initial_fill=32768, random_shuffle=True,
    prefetch_queue_depth=1, local_rank=0, name="reader",
    mean=(127.5, 127.5, 127.5), 
    std=(127.5, 127.5, 127.5)):
    """
    Parameters:
    ----------
    initial_fill: int
        Size of the buffer that is used for shuffling. If random_shuffle is False, this parameter is ignored.

    """
    rank: int = distributed.get_rank()
    world_size: int = distributed.get_world_size()
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.pipeline import Pipeline
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator

    pipe = Pipeline(
        batch_size=batch_size, num_threads=num_threads,
        device_id=local_rank, prefetch_queue_depth=prefetch_queue_depth, )
    condition_flip = fn.random.coin_flip(probability=0.5)
    with pipe:
        jpegs, labels = fn.readers.mxnet(
            path=rec_file, index_path=idx_file, initial_fill=initial_fill, 
            num_shards=world_size, shard_id=rank,
            random_shuffle=random_shuffle, pad_last_batch=False, name=name)
        images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
        images = fn.crop_mirror_normalize(
            images, dtype=types.FLOAT, mean=mean, std=std, mirror=condition_flip)
        pipe.set_outputs(images, labels)
    pipe.build()
    return DALIWarper(DALIClassificationIterator(pipelines=[pipe], reader_name=name, ))


@torch.no_grad()
class DALIWarper(object):
    def __init__(self, dali_iter):
        self.iter = dali_iter

    def __next__(self):
        data_dict = self.iter.__next__()[0]
        tensor_data = data_dict['data'].cuda()
        tensor_label: torch.Tensor = data_dict['label'].cuda().long()
        tensor_label.squeeze_()
        return tensor_data, tensor_label

    def __iter__(self):
        return self

    def reset(self):
        self.iter.reset()

if __name__ == "__main__":
    # root_dir = "/train/data/mask-ms1m-retinaface-t1"
    # root_dir = "/train/data/ms1m-retinaface-t1"
    root_dir = "/space/data/ms1m-retinaface-t1_v1"
    local_rank = 0
    dataset = MXFaceDataset(root_dir, local_rank)
    num = 10
    for i in range(num):
        dataset[i]
