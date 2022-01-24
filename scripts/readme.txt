::::::::::: SCRIPT INSTRUCTIONS ::::::::::::::::::
> self notes <


======== DATA PREPROCESSING ==============

- remember to set the

-. If plasticc dataset is in raw format, make sure to run:

    avocado_augment.py

- To produce the same data augmentation as used by our work, run:

    plasticc.preprocess_augment_v3.py

  this will produce a semi-balanced augmented dataset.

======== MMM-BOPF method ==============

-. To run MMM-BOPF parameter grid-search, ejecute:

    pre.pipeline.py

  this will find the optimal parameter configuration for our method and generate a config file.
  This process may take several hours depending on the machine power and was intended to be used only for
  thesis purpose. For further evaluations of this method, please use a custom config file provided with the code
  or write a different custom config file depending on user's needs.

-. to run MMM-BOPF feature representation technique, ejecute:

    fea.pipeline.py

  This will generate the feature representation of plasticc dataset in a h5 file.

-. To classify the MMM-BOPF representation using KNN-Classifier, ejecute:

    eval.classifier_knn.py

  This require to have the MMM-BOPF representation in h5 format.

-. To classify the MMM-BOPF representation using LGBM-Classifier, ejecute:

    eval.classifier_lgbm.py

  This require to have the MMM-BOPF representation in h5 format.

======== AVOCADO method ==============

-. To extract AVOCADO features, ejecute the avocado script (writen by AVOCADO's author):

    avocado_featurize.py

-. To Classify (train and predict) the AVOCADO features using KNN-classifier, ejecute:

    avocado_knn.py

-. To Train the AVOCADO features using LGBM-classifier, ejecute the avocado script (writen by AVOCADO's author):

    avocado_train_classifier.py

-. To predict the AVOCADO features using LGBM-classifier, ejecute the avocado script (writen by AVOCADO's author):

    avocado_predict.py

