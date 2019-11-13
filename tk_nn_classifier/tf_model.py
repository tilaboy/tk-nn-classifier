def build_model(model_cfg):
    """The train, save and evaluation if needed"""

    LOGGER.info('Loading: %s', model_cfg['embedding_file'])
    data_reader = DataReader(model_cfg['embedding_file'])

    LOGGER.info('Reading: %s', model_cfg['train_data'])
    _, x_train, y_train = data_reader.read_file(model_cfg['train_data'])

    LOGGER.info('Reading: %s', model_cfg['test_data'])
    _, x_test, y_test = data_reader.read_file(model_cfg['test_data'])

    input_dimension = x_train.shape[1]
    LOGGER.info('Input dimension: %i', input_dimension)

    model = build_graph(input_dimension)

    LOGGER.info("Start training")
    train(x_train, y_train, x_test, y_test, model, model_cfg)
    model.save(model_cfg['model_file'])

    test_loss, test_acc = model.evaluate(x_test, y_test)
    LOGGER.info("Test: loss {}\tacc {}".format(test_loss, test_acc))

    if 'devel_data' in model_cfg:
        _, x_devel, y_devel = data_reader.read_file(model_cfg['devel_data'])
        dev_loss, dev_acc = model.evaluate(x_devel, y_devel)
        LOGGER.info("Devel: loss {}\tacc {}".format(dev_loss, dev_acc))
