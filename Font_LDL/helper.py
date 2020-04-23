import sys
import time
import torch
import numpy as np

__all__ = ['Helper']


class Helper:
    checkpoint_history = []
    early_stop_monitor_vals = []
    best_score = 0
    best_epoch = 0

    def __init__(self):
        self.USE_GPU = torch.cuda.is_available()

    def checkpoint_model(self, model_to_save, optimizer_to_save, path_to_save, current_score, epoch, mode='min'):

        model_state = {'epoch': epoch + 1,
                       'model_state': model_to_save.state_dict(),
                       'score': current_score,
                       'optimizer': optimizer_to_save.state_dict()}

        # Save the model as a regular checkpoint
        torch.save(model_state, path_to_save + 'last.pth'.format(epoch))

        self.checkpoint_history.append(current_score)
        is_best = False

        # If the model is best so far according to the score, save as the best model state
        if ((np.max(self.checkpoint_history) == current_score and mode == 'max') or
                (np.min(self.checkpoint_history) == current_score and mode == 'min')):
            is_best = True
            self.best_score = current_score
            self.best_epoch = epoch
            # print('inside checkpoint', current_score, np.max(self.checkpoint_history))
            # torch.save(model_state, path_to_save + '{}_best.pth'.format(n_epoch))
            torch.save(model_state, path_to_save + 'best.pth')
            print('BEST saved at epoch: ')
            print("current score: ", current_score)
        if mode=="min":
            print('Current best', round(min(self.checkpoint_history), 7), 'after epoch {}'.format(self.best_epoch))
        else:
            print('Current best', round(max(self.checkpoint_history), 4), 'after epoch {}'.format(self.best_epoch))

        return is_best

    def load_saved_model(self, model, path, unsupervised =False):
        """
        Load a saved model from dump
        :return:
        """
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state'])
        if unsupervised:
            checkpoint['score']=10000
        print(">>>>>>>>>>>Loading model form epoch: ", checkpoint['epoch'])