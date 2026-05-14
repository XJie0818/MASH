import tools
from args import get_args
from dataloader import MASHdataloader
from MASH import *
from MASHtrain import Trainer

# 初始化
args = get_args()
logger = tools.init_logger()
tools.init_seed(256)


def main():
    logger.info('生成签到数据')
    poi_loader = MASHdataloader(args)
    if not args.dataset_test_path.exists():
        checkins = poi_loader.load(args.dataset, args.dataset_file)

    logger.info('生成训练数据')
    poi_loader.create_dataset(args.mode, args.dataset, drop_ratio=args.drop_ratio)
    config = poi_loader.config

    logger.info('加载')
    model = PoiModelnew(config)

    trainer = Trainer(config=config, logger=logger, gpu=config.gpu)
    if config.mode == 'train':
        trainer.train(model=model, dataloader=poi_loader)
    else:
        trainer.test(model, dataloader=poi_loader, model_path=config.model_path)


if __name__ == '__main__':
    main()


