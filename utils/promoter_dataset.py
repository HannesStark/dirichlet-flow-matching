import pandas as pd
import torch
import numpy as np
import pyBigWig
import tabix
from selene_sdk.targets import Target

from utils.selene_utils import MemmapGenome


class GenomicSignalFeatures(Target):
    """
    #Accept a list of cooler files as input.
    """

    def __init__(self, input_paths, features, shape, blacklists=None, blacklists_indices=None,
                 replacement_indices=None, replacement_scaling_factors=None):
        """
        Constructs a new `GenomicFeatures` object.
        """
        self.input_paths = input_paths
        self.initialized = False
        self.blacklists = blacklists
        self.blacklists_indices = blacklists_indices
        self.replacement_indices = replacement_indices
        self.replacement_scaling_factors = replacement_scaling_factors

        self.n_features = len(features)
        self.feature_index_dict = dict(
            [(feat, index) for index, feat in enumerate(features)])
        self.shape = (len(input_paths), *shape)

    def get_feature_data(self, chrom, start, end, nan_as_zero=True, feature_indices=None):
        if not self.initialized:
            self.data = [pyBigWig.open(path) for path in self.input_paths]
            if self.blacklists is not None:
                self.blacklists = [tabix.open(blacklist) for blacklist in self.blacklists]
            self.initialized = True

        if feature_indices is None:
            feature_indices = np.arange(len(self.data))

        wigmat = np.zeros((len(feature_indices), end - start), dtype=np.float32)
        for i in feature_indices:
            try:
                wigmat[i, :] = self.data[i].values(chrom, start, end, numpy=True)
            except:
                print(chrom, start, end, self.input_paths[i], flush=True)
                raise

        if self.blacklists is not None:
            if self.replacement_indices is None:
                if self.blacklists_indices is not None:
                    for blacklist, blacklist_indices in zip(self.blacklists, self.blacklists_indices):
                        for _, s, e in blacklist.query(chrom, start, end):
                            wigmat[blacklist_indices, np.fmax(int(s) - start, 0): int(e) - start] = 0
                else:
                    for blacklist in self.blacklists:
                        for _, s, e in blacklist.query(chrom, start, end):
                            wigmat[:, np.fmax(int(s) - start, 0): int(e) - start] = 0
            else:
                for blacklist, blacklist_indices, replacement_indices, replacement_scaling_factor in zip(
                        self.blacklists, self.blacklists_indices, self.replacement_indices,
                        self.replacement_scaling_factors):
                    for _, s, e in blacklist.query(chrom, start, end):
                        wigmat[blacklist_indices, np.fmax(int(s) - start, 0): int(e) - start] = wigmat[
                                                                                                replacement_indices,
                                                                                                np.fmax(int(s) - start,
                                                                                                        0): int(
                                                                                                    e) - start] * replacement_scaling_factor

        if nan_as_zero:
            wigmat[np.isnan(wigmat)] = 0
        return wigmat
class PromoterDataset(torch.utils.data.Dataset):
    def __init__(self, seqlength=1024, split="train", n_tsses=100000, rand_offset=0):
        self.shuffle = False

        class ModelParameters:
            seifeatures_file = 'data/promoter_design/target.sei.names'
            seimodel_file = 'data/promoter_design/best.sei.model.pth.tar'

            ref_file = 'data/promoter_design/Homo_sapiens.GRCh38.dna.primary_assembly.fa'
            ref_file_mmap = 'data/promoter_design/Homo_sapiens.GRCh38.dna.primary_assembly.fa.mmap'
            tsses_file = 'data/promoter_design/FANTOM_CAT.lv3_robust.tss.sortedby_fantomcage.hg38.v4.tsv'

            fantom_files = [
                "data/promoter_design/agg.plus.bw.bedgraph.bw",
                "data/promoter_design/agg.minus.bw.bedgraph.bw"
            ]
            fantom_blacklist_files = [
                "data/promoter_design/fantom.blacklist8.plus.bed.gz",
                "data/promoter_design/fantom.blacklist8.minus.bed.gz"
            ]

            n_time_steps = 400

            random_order = False
            speed_balanced = True
            ncat = 4
            num_epochs = 200

            lr = 5e-4

        config = ModelParameters()

        self.genome = MemmapGenome(
            input_path=config.ref_file,
            memmapfile=config.ref_file_mmap,
            blacklist_regions='hg38'
        )
        self.tfeature = GenomicSignalFeatures(
            config.fantom_files,
            ['cage_plus', 'cage_minus'],
            (2000,),
            config.fantom_blacklist_files
        )

        self.tsses = pd.read_table(config.tsses_file, sep='\t')
        self.tsses = self.tsses.iloc[:n_tsses, :]

        self.chr_lens = self.genome.get_chr_lens()
        self.split = split
        if split == "train":
            self.tsses = self.tsses.iloc[~np.isin(self.tsses['chr'].values, ['chr8', 'chr9', 'chr10'])]
        elif split == "valid":
            self.tsses = self.tsses.iloc[np.isin(self.tsses['chr'].values, ['chr10'])]
        elif split == "test":
            self.tsses = self.tsses.iloc[np.isin(self.tsses['chr'].values, ['chr8', 'chr9'])]
        else:
            raise ValueError
        self.rand_offset = rand_offset
        self.seqlength = seqlength

    def __len__(self):
        return self.tsses.shape[0]

    def __getitem__(self, tssi):
        chrm, pos, strand = self.tsses['chr'].values[tssi], self.tsses['TSS'].values[tssi], self.tsses['strand'].values[
            tssi]
        offset = 1 if strand == '-' else 0

        offset = offset + np.random.randint(-self.rand_offset, self.rand_offset + 1)
        seq = self.genome.get_encoding_from_coords(chrm, pos - int(self.seqlength / 2) + offset,
                                                   pos + int(self.seqlength / 2) + offset, strand)

        signal = self.tfeature.get_feature_data(chrm, pos - int(self.seqlength / 2) + offset,
                                                pos + int(self.seqlength / 2) + offset)
        if strand == '-':
            signal = signal[::-1, ::-1]
        return np.concatenate([seq, signal.T], axis=-1).astype(np.float32)

    def reset(self):
        np.random.seed(0)