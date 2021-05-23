import numpy as np
from collections import defaultdict
import avocado
from tqdm import tqdm


def select_real_objects_id(df_metadata, classes, n=50, ddf=1):
    wfd = df_metadata[df_metadata["ddf"] == ddf].copy()
    #     wfd["merged_class"] = [merged_labels_to_num[merged_labels[x]] for x in wfd["class"]]
    #     merged_classes = np.unique(wfd["merged_class"].to_numpy())
    #     print(merged_classes, classes)
    wfd_objects_id = None
    labels = None
    c_class = defaultdict(int)
    for c in classes:
        idxs = wfd[wfd["class"] == c].index.to_numpy()
        if len(idxs) > n:
            np.random.seed(0)
            idxs = np.random.choice(idxs, size=n, replace=False)

        c_class[c] = len(idxs)
        if wfd_objects_id is None:
            wfd_objects_id = idxs.copy()
            labels = np.full(len(idxs), c)
        else:
            wfd_objects_id = np.append(wfd_objects_id, idxs)
            labels = np.append(labels, np.full(len(idxs), c))

    return wfd_objects_id, labels, c_class


def add_augmented_versions_random(df_metadata, objs, c_class, n=50, ddf=1):
    wfd = df_metadata[df_metadata["ddf"] == ddf].copy()

    aug = None
    for c, v in c_class.items():
        if v < n:
            tmp = None
            df = wfd[wfd["class"] == c]
            for obj in objs:
                df2 = df[df["reference_object_id"] == obj]
                if len(df2) > 0:
                    values = df2.index.to_numpy()
                    if obj in values:
                        values = np.delete(values, np.where(values == obj)[0][0])
                    if len(values) > 0:
                        if tmp is None:
                            tmp = values.copy()
                        else:
                            tmp = np.append(tmp, values)
            m = n - v
            if len(tmp) > m:
                tmp = np.random.choice(tmp, size=m, replace=False)
            if len(tmp) > 0:
                if aug is None:
                    aug = tmp.copy()
                else:
                    aug = np.append(aug, tmp)

    return np.append(objs, aug)


def downscale_process_chunk(chunk, name, out_name, num_chunks, objects_id):
    dataset = avocado.load(
        name,
        chunk=chunk,
        num_chunks=num_chunks,
    )
    selected_objects = []
    for reference_object in tqdm(
            dataset.objects, desc="Object", dynamic_ncols=True
    ):
        if reference_object.metadata["object_id"] in objects_id:
            selected_objects.append(reference_object)

    if len(selected_objects) > 0:
        out_dataset = avocado.Dataset.from_objects(
            out_name,
            selected_objects,
            chunk=dataset.chunk,
            num_chunks=dataset.num_chunks,
        )

        out_dataset.write()


if __name__ == "__main__":
    print("loading set")
    # set_name = "plasticc_augment"
    # plasticc_augment_meta = avocado.load(set_name, metadata_only=True)
    set_name = "plasticc_test"
    plasticc_train_meta = avocado.load(set_name, metadata_only=True)

    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    print("selecting objs")
    objs1, labels, c_class = select_real_objects_id(plasticc_train_meta.metadata, classes, n=100, ddf=1)
    # print("adding augment")
    # objs2 = add_augmented_versions_random(plasticc_augment_meta.metadata, objs1, c_class, n=100, ddf=1)
    print("gen dataset")

    # num_chunks = 100
    # name = "plasticc_augment"
    # out_name = "plasticc_augment_ddf_100"
    # for chunk in tqdm(range(num_chunks), desc='Chunk',
    #                   dynamic_ncols=False):
    #     downscale_process_chunk(chunk, name, out_name, num_chunks, objs2)

    num_chunks = 200
    name = "plasticc_test"
    out_name = "plasticc_test_ddf_100"
    for chunk in tqdm(range(num_chunks), desc='Chunk',
                      dynamic_ncols=False):
        downscale_process_chunk(chunk, name, out_name, num_chunks, objs1)
