# System imports
import sys
import logging

def config_logging(verbose, log_file=None, append=False):
    log_format = '%(asctime)s %(levelname)s %(message)s'
    log_level = logging.DEBUG if verbose else logging.INFO
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(log_level)
    handlers = [stream_handler]
    if log_file is not None:
        file_mode = 'a' if append else 'w'
        file_handler = logging.FileHandler(log_file, mode=file_mode)
        file_handler.setLevel(log_level)
        handlers.append(file_handler)
    logging.basicConfig(level=log_level, format=log_format, handlers=handlers)