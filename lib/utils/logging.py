import os
import logging
import time
from pathlib import Path

def setup_logger(final_output_dir, rank, phase):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_rank{}.log'.format(phase, time_str, rank)
    final_log_file = os.path.join(final_output_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger, time_str

def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists() and cfg.RANK == 0:
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()
    else:
        while not root_output_dir.exists():
            print('=> wait for {} created'.format(root_output_dir))
            time.sleep(30)

    dataset = cfg.DATASET.DATASET
    dataset = dataset.replace(':', '_')
    model = cfg.MODEL.BACKBONE.NAME
    cfg_name = cfg.CFG_NAME
    # cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / model / cfg_name

    if cfg.RANK == 0:
        print('=> creating {}'.format(final_output_dir))
        final_output_dir.mkdir(parents=True, exist_ok=True)
        src_dir = os.path.join(final_output_dir, 'src')
        if not os.path.exists(src_dir): os.makedirs(src_dir)
    else:
        while not final_output_dir.exists():
            print('=> wait for {} created'.format(final_output_dir))
            time.sleep(5)

    return str(final_output_dir)