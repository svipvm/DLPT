# encoding: utf-8

import os, torch
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint, Timer
from ignite.contrib.handlers import ProgressBar

from utils.logger import get_current_logger
from utils.filesys import mkdir_if_not_exist


def do_train(cfg, model, train_loader, valid_loader, optimizer, loss_func):
    log_period = cfg.RECORD.LOG_PERIOD
    test_period = cfg.SOLVER.TEST_PERIOD
    model_dir = os.path.join(*[cfg.RECORD.OUTPUT_DIR, 'model'])
    device = 'cuda' if cfg.MODEL.DEVICES is not None else 'cpu'
    num_epochs = cfg.SOLVER.NUM_EPOCHS
    logger = get_current_logger(cfg)

    # logger_name = "-".join([cfg.MODEL.NAME, cfg.DATASETS.NAME])
    # from torch.optim import SGD
    # optimizer = SGD(model.parameters(), 0.1)
    trainer = create_supervised_trainer(model, optimizer, loss_func, device)

    metrics = {
        'accuracy': Accuracy(), 
        'ce_loss': Loss(loss_func)
    }
    evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

    # test_checkpointer = ModelCheckpoint(model_dir, 'cifar', test_period, n_saved=10, require_empty=False)
    # trainer.add_event_handler(Events.EPOCH_COMPLETED, test_checkpointer, {'model': model.state_dict()})
    # trainer.add_event_handler(Events.EPOCH_COMPLETED, test_checkpointer, {'model': model.state_dict(), 'optimizer': optimizer.state_dict()})

    timer = Timer(average=True)
    # timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
    #              pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'avg_loss')

    pbar = ProgressBar()
    pbar.attach(trainer) 

    @trainer.on(Events.EPOCH_STARTED)
    def log_before_train(engine):
        pass

    @trainer.on(Events.ITERATION_COMPLETED(every=log_period))
    def log_training_loss(engine):
        logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}".format(
            engine.state.epoch, 
            (engine.state.iteration - 1) % len(train_loader) + 1, 
            len(train_loader), 
            engine.state.metrics['avg_loss']))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        logger.info("Training Results - Epoch: {} Avg accuracy: {:.3f} Avg Loss: {:.3f}".format(
            engine.state.epoch, 
            evaluator.state.metrics['accuracy'], 
            evaluator.state.metrics['ce_loss']))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if valid_loader is None: return
        evaluator.run(valid_loader)
        logger.info("Validation Results - Epoch: {} Avg accuracy: {:.3f} Avg Loss: {:.3f}".format(
            engine.state.epoch, 
            evaluator.state.metrics['accuracy'], 
            evaluator.state.metrics['ce_loss']))

    @trainer.on(Events.EPOCH_COMPLETED(every=test_period))
    def save_model(engine):
        torch.save(model, mkdir_if_not_exist([model_dir, 'model_' + str(engine.state.epoch) + '.pt']))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_run_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'.format(
            engine.state.epoch, 
            timer.value() * timer.step_count,
            train_loader.batch_size / timer.value()))
        timer.reset()

    trainer.run(train_loader, num_epochs)
