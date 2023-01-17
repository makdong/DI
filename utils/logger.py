import logging
import logging.handlers
import os

# create the logger
logger = logging.getLogger()

# select the Level of logger
logger.setLevel(logging.INFO)

# format of log
formatter = logging.Formatter('%(levelname)s - %(message)s')

log_dir = './tests/logs/'
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

def set_logfile_name(filename):
    file_handler = logging.handlers.TimedRotatingFileHandler(filename = log_dir + filename, when='midnight', interval=1, encoding = 'utf-8')
    # file_handler = logging.FileHandler(filename = log_dir + filename, encoding = 'utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
