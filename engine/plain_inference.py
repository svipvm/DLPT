# encoding: utf-8
import logging

from ignite.engine import Events
from ignite.engine import create_supervised_evaluator
from ignite.metrics import Accuracy

from utils.logger import get_current_logger

def do_inference(cfg, model, test_loader):
    device = 'cuda' if cfg.MODEL.DEVICES is not None else 'cpu'

    logger = get_current_logger(cfg)
    logger.info("Start inferencing")
    evaluator = create_supervised_evaluator(model, metrics={'accuracy': Accuracy()}, device=device)

    @evaluator.on(Events.EPOCH_COMPLETED)
    def print_validation_results(engine):
        metrics = evaluator.state.metrics
        avg_acc = metrics['accuracy']
        logger.info("Validation Results - Accuracy: {:.3f}".format(avg_acc))

    evaluator.run(test_loader)
