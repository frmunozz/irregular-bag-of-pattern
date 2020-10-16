import numpy as np
from sklearn.model_selection import train_test_split


def get_small_dataset(dataset, times, labels, sort_class_count_index, classes, m, n1, n2, c):
    # n: size of the small dataset
    # c: use the 'c' classes most present in the original dataset

    classes_cut = []
    for i in sort_class_count_index[:c]:
        classes_cut.append(classes[i])

    data_time_tuple = []
    labels_cut = []
    for t, y, l in zip(times, dataset, labels):
        if l in classes_cut:
            data_time_tuple.append((t, y))
            labels_cut.append(l)

    x_train, x_test, y_train, y_test = train_test_split(data_time_tuple, labels_cut,
                                                        test_size=len(labels_cut) - n1,
                                                        random_state=0,
                                                        stratify=labels_cut)

    dataset_split = []
    times_split = []
    for d_tuple in x_train:
        times_split.append(d_tuple[0])
        dataset_split.append(d_tuple[1])

    _, x_test, _, y_test = train_test_split(x_test, y_test,
                                            test_size=n2,
                                            random_state=0,
                                            stratify=y_test)

    d_test_split = []
    t_test_split = []
    for d_tuple in x_test:
        d_test_split.append(d_tuple[1])
        t_test_split.append(d_tuple[0])

    return dataset_split, times_split, y_train, d_test_split, t_test_split, y_test


def gen_datasets(n1_arr, n2_arr, c_arr, datasets_folder):
    train_d_smalls = []
    train_t_smalls = []
    train_l_smalls = []
    test_d_smalls = []
    test_t_smalls = []
    test_l_smalls = []

    for c, n1, n2 in zip(c_arr, n1_arr, n2_arr):
        D_s1, t_s1, l_s1, D_s1_test, t_s1_test, l_s1_test = get_small_dataset(dataset, times, labels,
                                                                              sort_class_count_index, lclasses,
                                                                              m, n1, n2, c)
        train_d_smalls.append(D_s1)
        train_t_smalls.append(t_s1)
        train_l_smalls.append(l_s1)
        test_d_smalls.append(D_s1_test)
        test_t_smalls.append(t_s1_test)
        test_l_smalls.append(l_s1_test)

        # save to disk
        np.save(datasets_folder + "train_d_n{}_c{}.npy".format(n1, c), D_s1)
        np.save(datasets_folder + "train_t_n{}_c{}.npy".format(n1, c), t_s1)
        np.save(datasets_folder + "train_l_n{}_c{}.npy".format(n1, c), l_s1)
        np.save(datasets_folder + "test_d_n{}_c{}.npy".format(n2, c), D_s1_test)
        np.save(datasets_folder + "test_t_n{}_c{}.npy".format(n2, c), t_s1_test)
        np.save(datasets_folder + "test_l_n{}_c{}.npy".format(n2, c), l_s1_test)

        print(":::GEN train dataset of size:", n1, "and test dataset of size", n2, "with", c, "classes")

    return train_d_smalls, train_t_smalls, train_l_smalls, test_d_smalls, test_t_smalls, test_l_smalls
