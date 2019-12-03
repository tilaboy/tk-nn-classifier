tk\_nn\_classifier
==================

A framework package for text classification.

Internally, the package use two Deep Learning frameworks, ``SpaCy`` and
``Tensorflow``. ``Spacy`` packages come with some pre-trained models,
and can be easilly applied to proof of concept stories. In the ideal
work flow, Once the PoC works, one could try with more complicated
implementation, e.g. implemented ``Tensorflow`` DNN, RNN, CNN, or simple
CNN type perceptrons (Attention/Transformer will be added later),
together with in house word embeddings, with multiple feature inputs,
etc.

And the implementation of tensorflow models is fully compatiable with
``tf-serving``, which make the releasing and deployment of the new model
very simple.

And the nice part of the package is, everything you need to do is to add
more information into the config file.

Let's do it gradually together, step by step.

Installation
------------

::

    python setup.py develop

Tutorial
--------

1. simple start:
~~~~~~~~~~~~~~~~

Take the test config from ``cfg/spacy_test.cfg`` as example.

-  trxml\_fields: source files are trxml, and we need the following
   fields
-  features: contains input for the model
-  class: contains labels
-  doc\_id: a field which can be used to easily identify the document

-  datasets: input data

::

    {
        "trxml_fields": {
            "features": "sec_vacancy.0.sec_vacancy",
            "class": "derived_vac_intermediary.0.derived_vac_intermediary",
            "doc_id": "Document.0.correlationid"
        },

        "datasets": {
            "all_data": "tests/resource/samples/"
        }
    }

and then run:

``tk-nn-classifier train cfg/spacy_test.json``

and you will see the following logging message:

::

    I [2019-11-28 12:39:10,354] [tk_nn_classifier] use spacy spacy_poc
    I [2019-11-28 12:39:10,355] [tk_nn_classifier] starting training process ...
    I [2019-11-28 12:39:10,841] [tk_nn_classifier] Loaded model 'en_core_web_sm'
    I [2019-11-28 12:39:10,841] [tk_nn_classifier] split all_data into train and test
    I [2019-11-28 12:39:10,846] [tk_nn_classifier] split 10 docs into 8 train and 2 eval
    I [2019-11-28 12:39:10,846] [tk_nn_classifier] copy the train data to train folder models/poc/train
    I [2019-11-28 12:39:10,846] [tk_nn_classifier] copy the eval data to eval folder models/poc/eval
    I [2019-11-28 12:39:11,626] [xml_miner] reading trxml documents from dir models/poc/train
    I [2019-11-28 12:39:12,891] [xml_miner] reading trxml documents from dir models/poc/eval
    I [2019-11-28 12:39:13,649] [tk_nn_classifier] Training the model...
    LOSS    ACCU
    3.311   0.382
    2.909   0.541
    .... ....
    I [2019-11-28 12:41:11,778] [tk_nn_classifier] Saved model to models/poc

Congratulations, you just trained the first classifier. And the model is
save in ``models/poc`` by default, together with a label mapping file.

The package also support csv file as input, if you start with a csv
file, your first config file will looks like this:

::

    {
        "csv_fields": {
            "features": "full_text",
            "class": "class",
            "doc_id": "id"
        },

        "datasets": {
            "all_data": "tests/resource/samples.csv"
        }
    }

2. What happened behind
~~~~~~~~~~~~~~~~~~~~~~~

-  All data specified in the dataset is splitted into train and eval,
   and saved.

-  The learner read the feature input text, and tags from the train and
   eval

-  And derived a mapping from tag to an internal class\_id, save in the
   model path

-  Train the model for a number of epochs, and report the loss and
   accuracy on eval for each epoch

-  save the final model into the model path folder

3. A bit more config
~~~~~~~~~~~~~~~~~~~~

-  assume you want to work on different field as input, say
   ``derived_cond_contract_type``

-  and you want to train/eval from different data\_set, e.g.
   ``./customer_feedback``

-  and test on another data set, e.g. ``./another_data_set``

-  and you want to train on ``10`` epochs

-  and split you data with ratio ``train/eval = 7:3``

You will get a config like this:

::

    {
        "trxml_fields": {
            "features": "sec_vacancy.0.sec_vacancy",
            "class": "derived_cond_contract_type.0.derived_cond_contract_type",
            "doc_id": "Document.0.correlationid"
        },

        "num_epochs": 10,
        "split_ratio": 0.7,

        "datasets": {
            "all_data": "./customer_feedback",
            "test": {
                "eval_contract": "./another_data_set"
            }
        }
    }

run the train command ``tk-nn-classifier train cfg/spacy_test.json``

After the training, there will be a test running, since you specified a
test set. And write out the precision/recall and confusion matrix for
the prediction.

Example of precision/recall matrix:

::

    label                   Prec    Reca     F1
    Tijdelijk               0.714   0.385   0.500
    Vast                    0.575   0.676   0.622
    Unspecified           0.653 0.810   0.723
    Detachering / interim   1.000   0.333   0.500

Example of confusion matrix

::

    I [2019-11-28 12:41:14,672] [tk_nn_classifier] Confusion matrix:
    Predicted              Detachering / interim  Tijdelijk  Unspecified  Vast
    Actual
    Detachering / interim                      2          0            4     0
    Franchise                                  0          0            1     0
    Mogelijk vast                              0          0            0     1
    Tijdelijk                                  0         10           10     6
    Unspecified                                0          1           47    10
    Vast                                       0          3            8    23
    Vrijwilliger                               0          0            2     0

And also in ``models/poc``, you can also find the label mapping file, it
looks like this:
``{"0": "Detachering / interim", "1": "Franchise", "2": "Freelance", "3": "Mogelijk vast", "4": "Tijdelijk", "5": "Unspecified", "6": "Vast", "7": "Vrijwilliger"}``

4. use batch processing for test listed in test block
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also try the batch processing command to see which file get
wrong prediction:

``tk-nn-classifier predict cfg/spacy_test.json``

as default, the result is write to a csv file in the ``res`` folder, you
can also specify the output folder and which test to run if multiple
tests listed in the test\_sets block.

The output is a csv file, and it looks like this:

::

    Document.0.correlationid        new     old     probabilities
    02756a2de47d4e92875cc4d2007d9a83        Unspecified     Vast    {'Detachering / interim': 0.04226701334118843, 'Franchise': 0.11610068380832672, 'Freelance': 0.03618886321783066, 'Mogelijk vast': 0.08675872534513474, 'Tijdelijk': 0.08308562636375427, 'Unspecified': 0.27037227153778076, 'Vast': 0.2207392156124115, 'Vrijwilliger': 0.14448758959770203}
    03a3095422f44e6c9f2dcd04049a4a30        Vast    Vast    {'Detachering / interim': 0.00029618313419632614, 'Franchise': 0.0018887267215177417, 'Freelance': 0.0008601720910519361, 'Mogelijk vast': 0.0066970945335924625, 'Tijdelijk': 0.012247717939317226, 'Unspecified': 0.02053450606763363, 'Vast': 0.9532108306884766, 'Vrijwilliger': 0.004264758434146643}
    .....

5. more config
~~~~~~~~~~~~~~

You might give the model with different name for each version, and maybe
set to different path to export the model. Those could be set with the
follow configuration options:

::

    "model_type": "spacy_simple",
    "model_name": "vac_staffing_agency_spacy_v2",
    "model_path": "models/spacy_v2",

You can also add more training parameters, e.g. the dropout rate, number
of epochs:

::

    "dropout_rate": 0.2,
    "num_epochs": 20,

6. and more config
~~~~~~~~~~~~~~~~~~

There are many options one can add, e.g. Now you have train and eval
set, and you probably don't want to change it all the time, specially
when tweaking the RNN/CNN hyper-parameters.

You can specify them in your config. You can also specify more test
sets, as mentioned earlier, and label mapping files. For example:

::

    "datasets": {
        "train": "data_set/train_big",
        "eval": "data_set/eval_big",
        "test": {
            "unindentified": "test_set/unidentified/",
            "annotated": "test_set/annotated/",
            "eval": "test_set/eval",
            "random": "test_set/final_eval/random",
            "us": "test_set/us.csv"
        },
        "label_mapper": "models/label_mapper.json"
    }

Note that the data set can be eithor folder contains trxml/xml files, or
simply csv file. And you can just put them in any order in the test
block or as train, or eval data.

Just remember that for csv files, one needs another entry to tell the
learner which fields to take, similar to the trxml files. E.g.

::

    "csv_fields": {
        "features": "full_text",
        "class": "source_type",
        "doc_id": "posting_id",
        "extra": ["advertiser_name", "source_website", "source_url"]
    },

7. with tensorflow model
------------------------

Use tensorflow model should be as easy as switch the model type and add
the embedding file:

-  swith model type from 'spacy\_simple': currently, it support
   "tf\_cnn\_simple", "tf\_cnn\_multi", "tf\_lstm\_simple",
   "tf\_lstm\_multi", or "tf\_multi\_feat\_cnn". And still expanding
   (attention and transformer is on the way).

e.g. using a one lay cnn as feature extractor + fully connection layer

::

    "model_type": "tf_cnn_simple",
    "model_name": "staffing_agency_detector",
    "model_path": "models/tf/cnn_v3",

-  to add the embedding, add this block into your config:

::

     "embedding": {
         "file": "../../embeddings//en-cv.bin",
     },

If Tensforflow models are used, one can use all tf-serving tool to run
export the model, show model API, and run as service direct. e.g.

``saved_model_cli show --dir your_model_path --tag_set serve --signature_def serving_default``

will show infor for model input and output:

::

    The given SavedModel SignatureDef contains the following input(s):
      inputs['input'] tensor_info:
          dtype: DT_INT32
          shape: (-1, 512)
          name: input_text:0
    The given SavedModel SignatureDef contains the following output(s):
      outputs['classes'] tensor_info:
          dtype: DT_INT64
          shape: (-1)
          name: ArgMax:0
      outputs['probabilities'] tensor_info:
          dtype: DT_FLOAT
          shape: (-1, 2)
          name: softmax_tensor:0
    Method name is: tensorflow/serving/predict

8. even more config
-------------------

To be explained later:

::

    {
        "model_type": "tf_cnn_simple",
        "model_name": "staffing_agency_detector",
        "model_path": "models/tf/cnn",

        "dropout_rate": 0.5,
        "optimizer": "Adam",
        "learning_rate": 0.003,
        "num_epochs": 100,
        "batch_size": 128,
        "max_lines":50,

        "log_dir": "log",
        "max_sequence_length": 512,
        "max_steps_without_increase": 500,
        "min_train_steps": 1000,
        "check_per_steps": 100,

        "trxml_fields": {
            "features": "sec_vacancy.0.sec_vacancy",
            "class": "derived_vac_intermediary.0.derived_vac_intermediary",
            "doc_id": "Document.0.correlationid",
            "extra": ["derived_org_name.0.derived_org_name",
                "derived_source_site.0.derived_source_site",
                "derived_norm_url.0.derived_norm_url"]
        },

        "csv_fields": {
            "features": "full_text",
            "class": "source_type",
            "doc_id": "posting_id",
            "extra": ["advertiser_name", "source_website", "source_url"]
        },

        "lstm": {
            "hidden_size": 150,
            "nr_layers": 2
        },

        "cnn": {
            "nr_layers": 1,
            "filter_size": 32,
            "kernel_size": 3
        },

        "embedding": {
            "file": "../../embeddings/en-wv.bin",
            "dimension": 150,
            "token_encoding": "max_embedding",
            "trainable": false
        },

        "datasets": {
            "train": "data/train_big",
            "eval": "data/eval_big",
            "test": {
                "unindentified": "data/unidentified/",
                "annotated": "data/annotated/",
                "eval": "data/eval",
                "random": "data/final_eval/random",
                "us": "data/us.csv"
            },
            "label_mapper": "models/label_mapper.json"
        }
    }

Usage
-----

TRAIN: (example config\_file can be found in cfg/)

``tk-nn-classifier train config_file``

PROCESS BATCH:

``tk-nn-classifier predict config_file [output_folder] [test_set_name]``
