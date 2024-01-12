import os
import datetime
from functools import partial
import torch

join = partial(os.path.join)

# saveWeights
def saveWeights(train_acc, test_acc, epoch, model, optimizer,save_weights_path):
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict()}
    for root, dir, files in os.walk("IC_512"):
        if len(files) != 0:
            for file in files:
                if float(file.split('_')[
                             4]) < test_acc:  # test_acc小于当前acc的模型被替换，大于当前acc的模型仍然存在,如果当前acc比所有模型都好，那所有模型都会被删除，只剩当前最好的acc
                    os.remove(os.path.join(root, file))
                    save_file = save_weights_path + 'EEGImaginedCharacter_Transformer_%d_%.4f_%.4f_weights.pth' % (
                        epoch, train_acc, test_acc)
                    torch.save(state, save_file)
                    print('The model parameters have been saved successed')
        else:
            save_file = save_weights_path + 'EEGImaginedCharacter_Transformer_%d_%.4f_%.4f_weights.pth' % (
                epoch, train_acc, test_acc)
            torch.save(state, save_file)
            print('The model parameters have been saved successed')


def adjust_lr(epoch, optimizer, opt):
    new_lr = -1
    if epoch % opt.lr_decay_epoch == 0:
        new_lr = opt.init_lr * opt.lr_decay_eps**(epoch // opt.lr_decay_epoch)
    if new_lr != -1:
        for p in optimizer.param_groups:
            p['lr'] = new_lr


def print_to_log_file(log_path, *args, print_to_console=True, log_time=False):
    if log_time:
        args = (str(datetime.datetime.now()), *args)

    with open(log_path, 'a+') as f:
        for a in args:
            f.write(str(a))
            f.write('\t')
        f.write('\n')

    if print_to_console:
        print(*[str(a) + '\t' for a in args])


def mk_dirs(opt, subject):
    if opt.log_note is None:
        opt.log_note = str(datetime.datetime.now())
        opt.log_note = opt.log_note.replace(' ', '-').replace(':', '_').split('.')[0]
    opt.save_dir = join(opt.base_result_dir, subject + opt.model + f'_{opt.log_note}')
    opt.run_save_dir = join(opt.save_dir, 'run')
    opt.model_save_dir = join(opt.save_dir, 'models')
    opt.all_model_dir = join(opt.model_save_dir, 'all_model')
    opt.best_val_model_dir = join(opt.model_save_dir, 'best_acc')
    opt.latest_checkpoint_dir = join(opt.model_save_dir, 'latest_checkpoint')
    opt.config_save_path = join(opt.save_dir, 'config.json')
    opt.log_save_path = join(opt.save_dir, 'train_log.txt')

    os.makedirs(opt.save_dir)
    os.makedirs(opt.run_save_dir)
    os.makedirs(opt.model_save_dir)
    os.makedirs(opt.all_model_dir)
    os.makedirs(opt.best_val_model_dir)
    os.makedirs(opt.latest_checkpoint_dir)


def save_model(model, path, note, optimizer=None, opt=None, epoch=None, LR_Scheduler=None, for_conti_train=False):
    if for_conti_train:
        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'opt': opt, 'epoch': epoch,
                 'lrscheduler': LR_Scheduler.state_dict()}
    else:
        state = {'net': model.state_dict()}

    torch.save(state, os.path.join(path, note))


def save_and_clean_model(model, path, note, optimizer=None, opt=None, epoch=None, LR_Scheduler=None,
                         for_conti_train=False):
    for root, dir, files in os.walk(path):
        if len(files) != 0:
            for file in files:
                os.remove(os.path.join(root, file))
    save_model(model, path, note, optimizer, opt, epoch, LR_Scheduler, for_conti_train)
