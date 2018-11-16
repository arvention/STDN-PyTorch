import os
import os.path as osp
import numpy as np
import torch
import time
import datetime
import pickle
import torch.optim as optim
from utils.utils import to_var

from model import STDN
from loss.loss import get_loss
from layers.anchor_box import AnchorBox
from utils.timer import Timer

from data.pascal_voc import save_results as voc_save, do_python_eval


class Solver(object):

    DEFAULTS = {}

    def __init__(self, version, train_loader, test_loader, config):
        """
        Initializes a Solver object
        """

        super(Solver, self).__init__()
        self.__dict__.update(Solver.DEFAULTS, **config)
        self.version = version
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config

        self.build_model()

        # start with a pre-trained model
        if self.pretrained_model:
            self.load_pretrained_model()
        else:
            self.model.init_weights(self.model.multibox)

    def build_model(self):
        """
        Instantiate the model, loss criterion, and optimizer
        """

        # instatiate anchor boxes
        anchor_boxes = AnchorBox(map_sizes=[1, 3, 5, 9, 18, 36],
                                 aspect_ratios=self.aspect_ratios)
        self.anchor_boxes = anchor_boxes.get_boxes()

        # instatiate model
        self.model = STDN(mode=self.mode,
                          stdn_config=self.stdn_config,
                          channels=self.input_channels,
                          class_count=self.class_count,
                          anchor=self.anchor_boxes,
                          num_anchors=self.num_anchors)

        # instatiate loss criterion
        self.criterion = get_loss(config=self.config)

        # instatiate optimizer
        self.optimizer = optim.SGD(params=self.model.parameters(),
                                   lr=self.lr,
                                   momentum=self.momentum,
                                   weight_decay=self.weight_decay)

        # print network
        self.print_network(self.model, 'STDN')

        # use gpu if enabled
        if torch.cuda.is_available() and self.use_gpu:
            self.model.cuda()
            self.criterion.cuda()
            self.anchor_boxes.cuda()

    def print_network(self, model, name):
        """
        Prints the structure of the network and the total number of parameters
        """
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def load_pretrained_model(self):
        """
        loads a pre-trained model from a .pth file
        """
        self.model.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}.pth'.format(self.pretrained_model))))
        print('loaded trained model ver {}'.format(self.pretrained_model))

    def adjust_learning_rate(self, optimizer, gamma, step):
        """Sets the learning rate to the initial LR decayed by 10 at every
            specified step
        # Adapted from PyTorch Imagenet example:
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py
        """
        lr = self.lr * (gamma ** (step))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def print_loss_log(self,
                       start_time,
                       cur_iter,
                       class_loss,
                       loc_loss,
                       loss):
        """
        Prints the loss and elapsed time for each epoch
        """
        total_iter = self.num_iterations

        elapsed = time.time() - start_time
        total_time = (total_iter - cur_iter) * elapsed / (cur_iter + 1)

        total_time = str(datetime.timedelta(seconds=total_time))
        elapsed = str(datetime.timedelta(seconds=elapsed))

        log = "Elapsed {} -- {}, Iter [{}/{}]\n" \
              "class_loss: {:.4f}, loc_loss: {:.4f}, " \
              "loss: {:.4f}".format(elapsed,
                                    total_time,
                                    cur_iter + 1,
                                    total_iter,
                                    class_loss.item(),
                                    loc_loss.item(),
                                    loss.item())

        print(log)

    def save_model(self, i):
        """
        Saves a model per i iteration
        """
        path = os.path.join(
            self.model_save_path,
            '{}/{}.pth'.format(self.version, i + 1)
        )

        torch.save(self.model.state_dict(), path)

    def model_step(self, images, targets):
        """
        A step for each iteration
        """

        # empty the gradients of the model through the optimizer
        self.optimizer.zero_grad()

        # forward pass
        class_preds, loc_preds = self.model(images)

        # compute loss
        class_targets = [target[:, -1] for target in targets]
        loc_targets = [target[:, :-1] for target in targets]
        losses = self.criterion(class_preds=class_preds,
                                class_targets=class_targets,
                                loc_preds=loc_preds,
                                loc_targets=loc_targets,
                                anchors=self.anchor_boxes)
        class_loss, loc_loss, loss = losses

        # compute gradients using back propagation
        loss.backward()

        # update parameters
        self.optimizer.step()

        # return loss
        return class_loss, loc_loss, loss

    def train(self):
        """
        training process
        """

        # set model in training mode
        self.model.train()

        self.losses = []
        step_index = 0

        # start with a trained model if exists
        if self.pretrained_model:
            start = int(self.pretrained_model.split('/')[-1])
        else:
            start = 0

        # start training
        start_time = time.time()
        batch_iterator = iter(self.train_loader)
        for i in range(start, self.num_iterations):

            if i in self.sched_milestones:
                step_index += 1
                self.adjust_learning_rate(optimizer=self.optimizer,
                                          gamma=self.sched_gamma,
                                          step=step_index)

            try:
                images, targets = next(batch_iterator)
            except StopIteration:
                batch_iterator = iter(self.train_loader)
                images, targets = next(batch_iterator)

            images = to_var(images, self.use_gpu)
            targets = [to_var(target, self.use_gpu) for target in targets]

            class_loss, loc_loss, loss = self.model_step(images, targets)

            # print out loss log
            if (i + 1) % self.loss_log_step == 0:
                self.print_loss_log(start_time=start_time,
                                    cur_iter=i,
                                    class_loss=class_loss,
                                    loc_loss=loc_loss,
                                    loss=loss)
                self.losses.append([i, class_loss, loc_loss, loss])

            # save model
            if (i + 1) % self.model_save_step == 0:
                self.save_model(i)

        self.save_model(i)

        # print losses
        print('\n--Losses--')
        for i, class_loss, loc_loss, loss in self.losses:
            print(i, "{:.4f} {:.4f} {:.4f}".format(class_loss.item(),
                                                   loc_loss.item(),
                                                   loss.item()))

    def eval(self, dataset, top_k, threshold):
        num_images = len(dataset)
        all_boxes = [[[] for _ in range(num_images)]
                     for _ in range(self.class_count)]

        # timers
        _t = {'im_detect': Timer(), 'misc': Timer()}
        results_path = osp.join(self.result_save_path,
                                self.pretrained_model)
        det_file = osp.join(results_path,
                            'detections.pkl')

        for i in range(num_images):
            image, target, h, w = dataset.pull_item(i)

            image = to_var(image.unsqueeze(0), self.use_gpu)

            _t['im_detect'].tic()
            detections = self.model(image).data
            detect_time = _t['im_detect'].toc(average=False)

            # skip j = 0 because it is the background class
            for j in range(1, detections.shape[1]):
                dets = detections[0, j, :]
                mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
                dets = torch.masked_select(dets, mask).view(-1, 5)
                if dets.shape[0] == 0:
                    continue
                boxes = dets[:, 1:]
                boxes[:, 0] *= w
                boxes[:, 2] *= w
                boxes[:, 1] *= h
                boxes[:, 3] *= h
                scores = dets[:, 0].cpu().numpy()
                cls_dets = np.hstack((boxes.cpu().numpy(),
                                      scores[:, np.newaxis])).astype(np.float32,
                                                                     copy=False)
                all_boxes[j][i] = cls_dets

            print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,
                                                        num_images,
                                                        detect_time))

        with open(det_file, 'wb') as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

        print('Evaluating detections')

        if self.dataset == 'voc':
            voc_save(all_boxes, dataset, results_path)
            do_python_eval(results_path, dataset)

    def test(self):
        """
        testing process
        """
        self.model.eval()
        self.eval(dataset=self.test_loader.dataset,
                  top_k=5,
                  threshold=0.01)
