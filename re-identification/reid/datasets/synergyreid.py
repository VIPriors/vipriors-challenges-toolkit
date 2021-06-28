from __future__ import print_function, absolute_import
import os.path as osp
import numpy as np

from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json, read_json
from ..utils.data.dataset import _pluck


class SynergyReID(Dataset):
    md5 = '05050b5d9388563021315a81b531db7d'

    def __init__(self, root, split_id=0, num_val=100, download=True):
        super(SynergyReID, self).__init__(root, split_id=split_id)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. " +
                               "You can use download=True to download it.")

        self.load(num_val)

    def download(self):
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        import hashlib
        import shutil
        from glob import glob
        from zipfile import ZipFile

        raw_dir = osp.join(self.root, 'raw')
        mkdir_if_missing(raw_dir)

        # Open the raw zip file
        fpath = osp.join(raw_dir, 'synergyreid_data.zip')
        if osp.isfile(fpath) and \
           hashlib.md5(open(fpath, 'rb').read()).hexdigest() == self.md5:
            print("Using downloaded file: " + fpath)
        else:
            raise RuntimeError("Please move data to {} "
                               .format(fpath))

        # Extract the file
        exdir = osp.join(raw_dir, 'data_reid')
        if not osp.isdir(exdir):
            print("Extracting zip file")
            with ZipFile(fpath) as z:
                z.extractall(path=raw_dir)

        # Format
        images_dir = osp.join(self.root, 'images')
        mkdir_if_missing(images_dir)

        # 487 identities (+1 for background) with 2 camera views each
        # Here we use the convention that camera 0 is for query and
        # camera 1 is for gallery
        identities = [[[] for _ in range(2)] for _ in range(487)]

        def register(subdir):
            fpaths = sorted(glob(osp.join(exdir, subdir, '*.jpeg')))
            pids = set()
            for fpath in fpaths:
                fname = osp.basename(fpath)
                pid = int(fname.split('_')[0])
                cam = 1 if 'gallery' in subdir else 0
                pids.add(pid)
                fname = ('{:08d}_{:02d}_{:04d}.jpg'
                         .format(pid, cam, len(identities[pid][cam])))
                identities[pid][cam].append(fname)
                shutil.copy(fpath, osp.join(images_dir, fname))
            return pids

        trainval_pids = register('reid_training')
        gallery_val_pids = register('reid_val/gallery')
        query_val_pids = register('reid_val/query')

        assert query_val_pids <= gallery_val_pids
        assert trainval_pids.isdisjoint(query_val_pids)

        identities_test = [[[] for _ in range(2)] for _ in range(9172)]

        def register_test(subdir, n=0):
            fpaths = sorted(glob(osp.join(exdir, subdir, '*.jpeg')))
            print(len(fpaths))
            pids = set()
            for pindx, fpath in enumerate(fpaths):
                fname = osp.basename(fpath)
                pid = int(fname.split('.')[0])
                cam = 1 if 'gallery' in subdir else 0
                pids.add(pid)
                fname = ('{:08d}_{:02d}_{:04d}.jpg'
                         .format(pid, cam, 0))
                identities_test[pindx+n][cam].append(fname)
                shutil.copy(fpath, osp.join(images_dir, fname))
            return pids
        query_test_pids = register_test('reid_test/query')
        gallery_test_pids = register_test('reid_test/gallery',
                                          n=len(query_test_pids))

        # Save the training / val / test splits
        splits = [{
            'trainval': sorted(list(trainval_pids)),
            'query_val': sorted(list(query_val_pids)),
            'gallery_val': sorted(list(gallery_val_pids)),
            'query_test': sorted(list(query_test_pids)),
            'gallery_test': sorted(list(gallery_test_pids))}]
        write_json(splits, osp.join(self.root, 'splits.json'))

        # Save meta information into a json file
        meta = {'name': 'SynergyReID', 'shot': 'multiple', 'num_cameras': 2,
                'identities': identities, 'identities_test': identities_test}
        write_json(meta, osp.join(self.root, 'meta.json'))

    def load(self, verbose=True):
        splits = read_json(osp.join(self.root, 'splits.json'))
        if self.split_id >= len(splits):
            raise ValueError("split_id exceeds total splits {}"
                             .format(len(splits)))
        self.split = splits[self.split_id]

        trainval_pids = np.concatenate((np.asarray(self.split['trainval']),
                                       np.asarray(self.split['query_val'])))

        def _pluck_test(identities, indices, n=0):
            ret = []
            for index, pid in enumerate(indices):
                pid_images = identities[index+n]
                for camid, cam_images in enumerate(pid_images):
                    for fname in cam_images:
                        ret.append((fname, pid, camid))
            return ret

        self.meta = read_json(osp.join(self.root, 'meta.json'))
        identities = self.meta['identities']
        identities_test = self.meta['identities_test']
        self.train = _pluck(identities, self.split['trainval'], relabel=True)
        self.trainval = _pluck(identities, trainval_pids, relabel=True)
        self.query_val = _pluck(identities, self.split['query_val'])
        self.gallery_val = _pluck(identities, self.split['gallery_val'])
        self.query_test = _pluck_test(identities_test, self.split['query_test'])
        self.gallery_test = _pluck_test(identities_test, self.split['gallery_test'], n=len(self.split['query_test']))
        self.num_train_ids = len(self.split['trainval'])
        self.num_val_ids = len(self.split['query_val'])
        self.num_trainval_ids = len(trainval_pids)

        if verbose:
            print(self.__class__.__name__, "dataset loaded")
            print("  subset      | # ids | # images")
            print("  ---------------------------")
            print("  train       | {:5d} | {:8d}"
                  .format(self.num_train_ids, len(self.train)))
            print("  query val   | {:5d} | {:8d}"
                  .format(len(self.split['query_val']), len(self.query_val)))
            print("  gallery val | {:5d} | {:8d}"
                  .format(len(self.split['gallery_val']), len(self.gallery_val)))
            print("  trainval    | {:5d} | {:8d}"
                  .format(self.num_trainval_ids, len(self.trainval)))
            print("  ---------------------------")
            print("  query test  | {:5d} | {:8d}"
                  .format(len(self.split['query_test']), len(self.query_test)))
            print("  gallery test | {:5d} | {:8d}"
                  .format(len(self.split['gallery_test']), len(self.gallery_test)))
