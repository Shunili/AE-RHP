# System
import os
import argparse
import logging

# Externals
import yaml
import numpy as np

# Locals
from datasets import get_data_loaders
from trainers import get_trainer
from utils.config_logging import config_logging

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/MP_n=4.yaml',
            help='YAML configuration file')

    add_arg('--gpu', type=int, default=None,
            help="Option for local tasks.")
    add_arg('--resume', action='store_true',
            help='Resume training from last checkpoint')
    add_arg('-v', '--verbose', action='store_true',
            help='Enable verbose logging')
  
    return parser.parse_known_args()[0]

def load_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def main():
    """Main function"""

    # Initialization
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Prepare output directory
    output_dir = config.get('output_dir', None)
    if output_dir is not None:
        output_dir = os.path.expandvars(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    # Setup logging
    log_file = (os.path.join(output_dir, 'out.log')
                if output_dir is not None else None)
    config_logging(verbose=args.verbose, log_file=log_file, append=args.resume)
    logging.info('Configuration: %s' % config)

    # Load the datasets
    logging.info('Createing dataloaders...')
    train_data_loader, valid_data_loader = get_data_loaders(distributed=False, **config['data'])
    logging.info('Dataloaders done!')

    # Load the trainer
    if args.gpu is not None:
        logging.info('Using GPU %i', args.gpu)
        
    logging.info('Building trainer...')
    trainer = get_trainer(name=config['trainer'], output_dir=output_dir, gpu=args.gpu)

    # Build the model and optimizer
    trainer.build(config)
    logging.info('Trainer is successfullt built.')

    # Resume from checkpoint
    if args.resume:
        trainer.load_checkpoint()

    # Run the training
    logging.info('Start training...')
    summary = trainer.train(train_data_loader=train_data_loader,
                            valid_data_loader=valid_data_loader,
                            **config['train'])

    # Print some conclusions
    # try_barrier()
    n_train_samples = len(train_data_loader.sampler)
    logging.info('Finished training')
    train_time = np.mean(summary['train_time'])
    logging.info('Train samples %g time %g s rate %g samples/s',
                 n_train_samples, train_time, n_train_samples / train_time)
    if valid_data_loader is not None:
        n_valid_samples = len(valid_data_loader.sampler)
        valid_time = np.mean(summary['valid_time'])
        logging.info('Valid samples %g time %g s rate %g samples/s',
                     n_valid_samples, valid_time, n_valid_samples / valid_time)

    logging.info('All done!')

if __name__ == '__main__':
    main()