# Attention: GELU激活函数要求torch version >= 1.4.0

# -*-coding:utf-8-*-
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from data_preocess import getCharacter_data,MyDataSet
import numpy as np
import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
from common_utils import  print_to_log_file, saveWeights
import datetime
from tensorboardX import SummaryWriter
from model import Transformer
import argparse
from config import get_config



# learning_rate_decay
def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 60 epochs"""
    if epoch > 100:
        lr *= (0.9 ** (epoch // 30))
    else:
        lr *= (0.98 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# IC_log
log_note = None
if log_note is None:
    log_note = str(datetime.datetime.now())
    log_note = log_note.replace(' ', '-').replace(':', '_').split('.')[0]
log_save_path = "./IC_log/" + log_note
writer = SummaryWriter(log_dir="./tensorboard_Character_logs", flush_secs=2)


def main(config):
    enc_inputs, dec_outputs, enc_inputs_test, dec_outputs_test = getCharacter_data(config.data_path)

    # 封装loader
    loader = Data.DataLoader(MyDataSet(enc_inputs, dec_outputs), config.batch_size_train, True)
    loader_test = Data.DataLoader(MyDataSet(enc_inputs_test, dec_outputs_test), config.batch_size_test, True)
    # 定义模型和优化器
    model = Transformer(config).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=7e-5, betas=(0.9, 0.999), eps=1e-8)
    lr_init = optimizer.param_groups[0]['lr']



    for epoch in range(config.epochs):
        train_correct = 0  # 训练正确的次数
        train_total = 0
        train_loss_sum = 0

        test_correct = 0  # 训练正确的次数
        test_total = 0
        test_loss_sum = 0

        batch_id = 0
        batch_id_test = 0
        adjust_learning_rate(optimizer, epoch, lr_init)
        learning_rate = optimizer.param_groups[0]['lr']

        for enc_inputs, dec_outputs in loader:
            model.train()
            enc_inputs, dec_outputs = enc_inputs.cuda(), dec_outputs.cuda()
            class_res, enc_self_attns = model(enc_inputs)

            # loss
            dec_outputs = dec_outputs.squeeze(2)
            pre_outputs = class_res.view(-1, class_res.size(-1))
            loss = criterion(pre_outputs, dec_outputs.view(-1))

            # acc
            label_acc = dec_outputs.squeeze(1).cpu().numpy()
            output_acc = torch.argmax(class_res, 1).cpu().numpy()
            predict_acc = output_acc
            predict_acc = torch.LongTensor(predict_acc)
            label_acc = torch.LongTensor(label_acc)

            train_total += label_acc.size(0)
            train_correct += (predict_acc == label_acc).sum()  # 获取训练正确地次数
            train_loss_sum += loss.item()
            batch_id += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_acc_0 = 100 * train_correct.double() / train_total
        train_acc = train_acc_0.cpu().numpy()  # train_acc_0是一个tensor数组，需要先转为cpu.tensor，再转为numpy
        train_loss = train_loss_sum / batch_id

        # # load weight
        # checkpoint = torch.load(config.finish_weights_path)
        # model.load_state_dict(checkpoint['net'])

        for enc_inputs_test, dec_outputs_test in loader_test:
            model.eval()
            enc_inputs_test, dec_outputs_test = enc_inputs_test.cuda(), dec_outputs_test.cuda()
            class_res_test, enc_self_attns_test = model(enc_inputs_test)

            # loss
            dec_outputs_test = dec_outputs_test.squeeze(2)
            pre_outputs = class_res_test.view(-1, class_res_test.size(-1))
            loss_test = criterion(pre_outputs, dec_outputs_test.view(-1))

            label_test_acc = dec_outputs_test.squeeze().cpu().numpy()
            output_acc = torch.argmax(class_res_test, 1).cpu().numpy()
            predict_test_acc = output_acc
            predict_test_acc = torch.LongTensor(predict_test_acc)
            label_test_acc = torch.LongTensor(label_test_acc)

            test_total += label_test_acc.size(0)
            test_loss_sum += loss_test.item()
            batch_id_test += 1
            test_correct += (predict_test_acc == label_test_acc).sum()  # 获取训练正确地次数

        test_acc_0 = 100 * test_correct.double() / test_total
        test_acc = test_acc_0.cpu().numpy()
        test_loss = test_loss_sum / batch_id_test

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/test', test_acc, epoch)

        print('Epoch:', '%04d' % (epoch + 1), 'train_loss =', '{:.6f}'.format(train_loss), 'train_acc =',
              ('%.4f' % train_acc), 'test_loss =', ('%.6f' % test_loss), 'test_acc =', ('%.4f' % test_acc),
              'learning_rate =', ('%.8f' % learning_rate))
        print_to_log_file(log_save_path, 'Epoch:', '%04d' % (epoch + 1), 'train_loss =', ('%.6f' % train_loss),
                          'train_acc =',
                          ('%.4f' % train_acc), 'test_loss =', ('%.6f' % test_loss), 'test_acc =', ('%.4f' % test_acc),
                          print_to_console=False)

        # save model weights
        if test_acc >= 95.27:
            saveWeights(train_acc, test_acc, epoch + 1, model, optimizer,config.save_weights_path)

        # record the best train accuracy and best train test accuracy and best test accuracy
        if epoch == 0:
            best_accuracy = test_acc
            best_train_accuracy = train_acc
            best_test_train_accuracy = train_acc
            best_epoch = epoch
            stop_count = 0
        else:
            if best_accuracy <= test_acc:
                best_accuracy = test_acc
                best_test_train_accuracy = train_acc
                best_epoch = epoch
                stop_count = 0
            else:
                stop_count += 1
        best_train_accuracy = max(train_acc, best_train_accuracy)
    # print the highest accuracy as the best accuracy
    print(
        'best epoch : %d/%d   best train accuracy : %f%%   best test train accuracy : %f%%   best test accuracy : %f%%' \
        % (best_epoch + 1, config.epochs, best_train_accuracy, best_test_train_accuracy, best_accuracy))

    print_to_log_file(log_save_path, f"best_epoch: {best_epoch + 1}    "
                                     f"best_train_acc:{best_train_accuracy:.4f}    "
                                     f"best_test_train_acc:{best_test_train_accuracy:.4f}    "
                                     f"best_test_acc:{best_accuracy:.4f}", print_to_console=False)


writer.close()
def update_config(args, config):
    for attr in config.__dict__:
        if hasattr(args, attr):
            if getattr(args, attr) != None:
                setattr(config, attr,getattr(args,attr))

    return config


def get_parser():
    #  表示在命令行显示帮助信息的时候，这个程序的描述信息
    parser = argparse.ArgumentParser(description="Train and test")
    # add_argument函数来增加参数。default参数表示默认值

    return parser

if __name__ == '__main__':

    import time
    torch.cuda.synchronize()  # 增加同步操作
    start = time.time()
    args = get_parser()
    args = args.parse_args()
    config = get_config()
    config = update_config(args,config)
    main(config)
    torch.cuda.synchronize()  # 增加同步操作
    end = time.time()
    elapsed_time = end - start
    print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))




