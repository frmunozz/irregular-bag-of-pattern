import os.path

from ibopf.settings import settings, get_data_directory
from ibopf.avocado_adapter import Dataset, MMMBOPFFeaturizer, AVOCADOFeaturizer
from ibopf import IBOPF, CompactIBOPF
import umap
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
from tqdm import tqdm
import random

def load_features(dataset_name, method="IBOPF", tag="features_v3_LSA", sparse=False, selected_classes=None):
    # Load the dataset (metadata only)
    print("Loading dataset '%s'..." % dataset_name)
    dataset = Dataset.load(dataset_name, metadata_only=True)
    dataset.set_method(method)

    # Load the dataset features depending on method
    print("Loading features...")
    if method == "IBOPF":
        if sparse:
            dataset.load_sparse_features(tag)
        else:
            dataset.load_compact_features(tag)
        featurizer = MMMBOPFFeaturizer(include_metadata=True)
    else:
        dataset.load_raw_features()
        featurizer = AVOCADOFeaturizer()

    df_features = dataset.select_features(featurizer)
    metadata = dataset.metadata

    if metadata.shape[0] != df_features.shape[0]:
        # print("reducing metadata")
        metadata = metadata[metadata.index.isin(df_features.index)]

    if selected_classes is not None:
        metadata = metadata[metadata["class"].isin(selected_classes)]
        df_features = df_features[df_features.index.isin(metadata.index)]
    labels = metadata["class"]
    # print(np.unique(labels))
    return df_features, labels


plot_labels_extra_short = {
    6: 'Single microlens',
    15: 'TDE',
    16: 'Eclip. Binary',
    42: 'SNII',
    52: 'SNIax',
    53: 'Mira',
    62: 'SNIbc',
    64: 'Kilonova',
    65: 'M-dwarf',
    67: 'SNIa-91bg',
    88: 'AGN',
    90: 'SNIa',
    92: 'RR lyrae',
    95: 'SLSN-I',
    99: 'Unknown',
}


def prepare_labels(labels, idxs=None):
    try:
        labels = labels.to_numpy()
    except:
        pass
    if idxs is not None:
        labels = labels[idxs]
    labels_verbose = [plot_labels_extra_short[x] for x in labels]
    # print(np.unique(labels_verbose))
    le = LabelEncoder()
    le.fit(np.unique(labels_verbose))
    y = le.transform(labels_verbose)
    return y, le

def main(fit_with_labels):
    dataset = "plasticc_augment_v3"
    method="AVOCADO"
    tag="features_v1"
    fea, labels = load_features(
        dataset, method=method, tag=tag)

    reducer = umap.UMAP(n_components=2, n_neighbors=100, verbose=True, random_state=42,
                            min_dist=0.9, metric="cosine", densmap=False, n_jobs=-1)
    columns = None
    if any(fea.isna().any()):
        columns = fea.columns[fea.isna().any()].tolist()
        fea = fea.drop(columns=columns)
    scaler = StandardScaler()
    fea = scaler.fit_transform(fea)

    if fit_with_labels:
        fea_red = reducer.fit_transform(fea, y=labels)
    else:
        fea_red = reducer.fit_transform(fea)

    try:
        fea_red = fea_red.values
    except:
        pass
    base_path = get_data_directory()
    filename = "AVOCADO_UMAP"
    if fit_with_labels:
        filename += "_supervised"
    output = os.path.join(base_path, filename + "%s_2d.npy" % dataset)
    print("writing file", output)
    np.save(output, fea_red)

    test_fea = None
    dataset_name = "plasticc_test"
    for chunk in tqdm(range(25), desc='chunks',
                      dynamic_ncols=True):
        dataset = Dataset.load(dataset_name, metadata_only=True, chunk=chunk, num_chunks=25)
        dataset.set_method("AVOCADO")
        dataset.load_raw_features(tag="features_v1")
        avocado_fea = dataset.select_features(AVOCADOFeaturizer(discard_metadata=False))
        labels = dataset.metadata["class"].to_numpy()

        if columns is not None:
            avocado_fea = avocado_fea.drop(columns=columns)

        avocado_fea = avocado_fea.values


        random.seed(chunk)  # directly use the current chunk num as seed, should be ok
        idxs = random.sample(range(avocado_fea.shape[0]), avocado_fea.shape[0] // 10)
        avocado_fea = avocado_fea[idxs]
        labels = labels[idxs]

        avocado_fea = scaler.transform(avocado_fea)

        if fit_with_labels:
            fea_red = reducer.fit_transform(avocado_fea, y=labels)
        else:
            fea_red = reducer.fit_transform(avocado_fea)

        try:
            fea_red = fea_red.values
        except:
            pass

        if test_fea is None:
            test_fea = fea_red
        else:
            test_fea = np.vstack((test_fea, fea_red))

    np.save(os.path.join(base_path, filename + "%s_2d.npy" % dataset_name), test_fea)


if __name__ == '__main__':
    # main(False)
    main(True)
