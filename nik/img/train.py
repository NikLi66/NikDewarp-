import argparse
import logging
import os
import time

import torch
import torch.utils.data as Data
from tensorboardX import SummaryWriter

from nik.img.dataloader import NikDataSet
from nik.img.loss import NikLosses

from nik.utils.util_logger import logging_to_file
from nik.img.unet import ImgUNet
from nik.img.restormer import Restormer
from nik.utils.util_lr import CustomLRScheduler
from nik.utils.util_logger import collect_monitor, string_monitor, average_monitor


class Trainer:
    def __init__(self, args):
        # log
        logging_to_file(os.path.join(args.output_dir, "log.txt"))
        self._logger = logging.getLogger()
        self._writer = SummaryWriter(args.output_dir)
        self._logger.info(args)
        os.system("git log -n 1 > %s/git_info" % args.output_dir)


        # gpu / cpu
        if torch.cuda.is_available():
            self._logger.info('train with gpu and pytorch {}'.format(torch.__version__))
            self._device = torch.device("cuda")
        else:
            self._logger.info('train with cpu and pytorch {}'.format(torch.__version__))
            self._device = torch.device("cpu")

        #model
        in_channels = 1
        if args.use_canny_input == 2:
            in_channels = in_channels * 2
        if args.use_ori == 1:
            in_channels = in_channels * 2

        if args.model_name == 'restormer':
            self._model = Restormer(inp_channels=in_channels, out_channels=2, dim=32)
        else:
            self._model = ImgUNet(in_channels=in_channels, out_channels=2, use_ori=args.use_ori, use_canny_input=args.use_canny_input, model_name=args.model_name)

        # DDP
        # torch.distributed.init_process_group(backend='nccl')
        # rank = torch.distributed.get_rank()
        # self._device = torch.device('cuda:{}'.format(rank))
        # torch.cuda.set_device(self._device)
        # self._model.cuda()
        # self._model = torch.nn.parallel.DistributedDataParallel(self._model, find_unused_parameters=False)

        # data
        self._train_data = NikDataSet(img_height=args.img_height, img_width=args.img_width, pk_list_path=args.train_list_path, use_ori=args.use_ori, use_canny_input=args.use_canny_input,num_workers=60)
        self._train_loader = Data.DataLoader(dataset=self._train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
        self._train_steps = len(self._train_loader)

        self._test_data = NikDataSet(img_height=args.img_height, img_width=args.img_width, pk_list_path=args.test_list_path, use_ori=args.use_ori, use_canny_input=args.use_canny_input,num_workers=20)
        self._test_loader = Data.DataLoader(dataset=self._test_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
        self._test_steps = len(self._test_loader)


        self._model.print_param()
        self._model = self._model.to(self._device)
        self._model = torch.nn.DataParallel(self._model, device_ids=[i for i in range(args.num_gpus)])

        # loss
        self._criterion = NikLosses(device=self._device, img_height=args.img_height, img_width=args.img_width)

        # optimizer
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=args.learning_rate)
        self._scheduler = CustomLRScheduler(optimizer=self._optimizer, init_lr=args.learning_rate,  warmup_steps=args.warm_up_step)

    def load_checkpoint(self, checkpoint_path):
        state = torch.load(checkpoint_path)
        self._model.load_state_dict(state['state_dict'])
        self._logger.info('model loaded from %s' % checkpoint_path)
        return state['epoch']

    def save_checkpoint(self, checkpoint_path, epoch):
        state = {'state_dict': self._model.state_dict(),
                 'epoch': epoch}
        torch.save(state, checkpoint_path)
        self._logger.info('model saved to %s' % checkpoint_path)

    def run_train(self, epoch):
        self._model.train()
        self._optimizer.zero_grad()

        self._scheduler.global_step = epoch * self._train_steps
        self._scheduler.step()

        start = time.time()
        train_batch_monitor = None
        train_total_monitor = None
        for i, (img, fp, mask) in enumerate(self._train_loader):
            # print(img.shape, fp.shape, ori.shape, mask.shape)
            # Forward BCHW
            if args.use_ori == 0:
                o, o0, o1, o2, o3 = self._model(img.to(self._device))
            else:
                input = torch.concatenate([img, ori], dim=1)
                o, o0, o1, o2, o3 = self._model(input.to(self._device))
            # print(o.shape, o0.shape, o1.shape, o2.shape, o3.shape)
            l, l0, l1, l2, l3 = self._criterion.multi_loss(o, o0, o1, o2, o3, fp.to(self._device), mask.to(self._device))
            loss = l + 0.01 * l0 + 0.01 * l1 + 0.01 * l2 + 0.01 * l3

            # 更新lr
            self._scheduler.step()
            # 计算梯度
            loss.backward()
            # 梯度保护
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
            # 更新参数
            self._optimizer.step()
            # 清空梯度
            self._optimizer.zero_grad()

            # 计算统计量
            cur_monitor = {"loss": loss.item(), "l": l.item(), "l0": l0.item(), "l1": l1.item(), "l2": l2.item(), "l3": l3.item()}
            train_batch_monitor = collect_monitor(train_batch_monitor, cur_monitor)
            train_total_monitor = collect_monitor(train_total_monitor, cur_monitor)

            if ((i + 1) % args.display_interval) == 0:
                # 打印信息
                batch_time = time.time() - start
                train_batch_monitor = average_monitor(train_batch_monitor, args.display_interval)
                self._logger.info(
                    '[%d], [%d/%d], %s, time:%0.4f, lr:%0.4f' % (
                        epoch + 1, (i + 1), self._train_steps, string_monitor(train_batch_monitor), batch_time,
                        self._scheduler.cur_lr))
                start = time.time()
                train_batch_monitor = None

            # add log
            cur_step = epoch * self._train_steps + i
            self._writer.add_scalar(tag='Train/loss', scalar_value=loss.item(), global_step=cur_step)
            self._writer.add_scalar(tag='Train/lr', scalar_value=self._scheduler.cur_lr, global_step=cur_step)

        return average_monitor(train_total_monitor, self._train_steps)

    def run_eval(self, epoch):
        self._model.eval()
        start = time.time()
        test_batch_monitor = None
        test_total_monitor = None
        for i, (img, fp, mask) in enumerate(self._test_loader):
            # Forward BCHW
            if args.use_ori == 0:
                o, o0, o1, o2, o3 = self._model(img.to(self._device))
            else:
                input = torch.concatenate([img, ori], dim=1)
                o, o0, o1, o2, o3 = self._model(input.to(self._device))

            l, l0, l1, l2, l3 = self._criterion.multi_loss(o, o0, o1, o2, o3, fp.to(self._device), mask.to(self._device))
            loss = l + 0.01 * l0 + 0.01 * l1 + 0.01 * l2 + 0.01 * l3

            # 计算统计量
            cur_monitor = {"loss": loss.item(), "l": l.item(), "l0": l0.item(), "l1": l1.item(), "l2": l2.item(), "l3": l3.item()}
            test_batch_monitor = collect_monitor(test_batch_monitor, cur_monitor)
            test_total_monitor = collect_monitor(test_total_monitor, cur_monitor)

            if ((i + 1) % args.display_interval) == 0:
                # 打印信息
                batch_time = time.time() - start
                test_batch_monitor = average_monitor(test_batch_monitor, args.display_interval)
                self._logger.info(
                    '[%d], [%d/%d], %s, time:%0.4f, lr:%0.4f' % (
                        epoch + 1, (i + 1), self._test_steps, string_monitor(test_batch_monitor), batch_time,
                        self._scheduler.cur_lr))
                start = time.time()
                test_batch_monitor = None

        return average_monitor(test_total_monitor, self._test_steps)


def run(args):
    trainer = Trainer(args)
    start_epoch = 0
    epoch = 0

    try:
        # if exist checkpoint ,calc loss first
        if os.path.exists(args.checkpoint):
            start_epoch = trainer.load_checkpoint(args.checkpoint)
            start = time.time()
            test_total_monitor = trainer.run_eval(start_epoch)
            logging.info('[%d/%d], |Eval:| %s, time:%0.4f' % (start_epoch, args.epochs, string_monitor(test_total_monitor), time.time() - start))


        for epoch in range(start_epoch, args.epochs):
            start = time.time()
            train_total_monitor = trainer.run_train(epoch)
            logging.info('[%d/%d], |Train:| %s, time:%0.4f, lr:%0.4f' % (epoch, args.epochs, string_monitor(train_total_monitor), time.time() - start, trainer._scheduler.cur_lr))
            checkpoint_path = '{}/TalUnet_{}_{}.pth'.format(args.output_dir, epoch + 1, train_total_monitor["loss"])
            trainer.save_checkpoint(checkpoint_path, epoch)

            final_path = '{}/final.pth'.format(args.output_dir)
            os.system('ln -sf {} {}'.format(checkpoint_path, final_path))

    except KeyboardInterrupt:
        trainer.save_checkpoint('{}/final.pth'.format(args.output_dir), epoch)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # globla config
    parser.add_argument('--train_list_path', type=str,
                        default='./train.pk.lst',
                        help='train pk lst path')
    parser.add_argument('--test_list_path', type=str,
                        default='./train.pk.lst',
                        help='test pk lst path')
    parser.add_argument('--output_dir', type=str,
                        default='./output/',
                        help='output dir')
    parser.add_argument('--checkpoint', type=str,
                        default='',
                        help='init')
    parser.add_argument('--learning_rate', type=float,
                        default=0.1,
                        help='learning rate')
    parser.add_argument('--warm_up_step', type=int,
                        default=5000,
                        help='warm up step')
    parser.add_argument('--display_interval', type=int,
                        default=10,
                        help='display interval')
    parser.add_argument('--epochs', type=int,
                        default=300,
                        help='total epochs')
    parser.add_argument('--batch_size', type=int,
                        default=32,
                        help='train batch size')
    parser.add_argument('--img_height', type=int,
                        default=512,
                        help='img height')
    parser.add_argument('--img_width', type=int,
                        default=512,
                        help='img width')
    parser.add_argument('--use_ori', type=int,
                        default=1,
                        help='using ori')
    parser.add_argument('--use_canny_input', type=int,
                        default=0,
                        help='using canny input')
    parser.add_argument('--model_name', type=str,
                        default='unet3',
                        help='model name')
    parser.add_argument('--use_transform', type=int,
                        default=0,
                        help='using transform')
    parser.add_argument('--debug', type=int,
                        default=0,
                        help='debug flag')
    parser.add_argument('--num_gpus', type=int,
                        default=1,
                        help='number of gpus')

    args, _ = parser.parse_known_args()
    if args.debug > 0:
        args.train_list_path = "./test.lst"
        args.test_list_path = "./test.lst"
        args.output_dir = "./debug/"
        args.display_interval = 1
    run(args)
