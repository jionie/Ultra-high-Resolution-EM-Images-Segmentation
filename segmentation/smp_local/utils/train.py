import sys
import torch
from tqdm import tqdm as tqdm
from .meter import AverageValueMeter
from .lr_scheduler import OneCycleScheduler, CosineAnnealingWarmUpRestarts
from apex import amp

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
        # m.requires_grad_(False)
        
# optim = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),lr=schedule[0])

# def fix_bn(m):
#     if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
#         m.eval()        

class Epoch:

    def __init__(self, model, loss, metrics, stage_name, device='cpu', verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for x, y in iterator:
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainEpoch(Epoch):

    def __init__(self, model, loss, metrics, optimizer, device='cpu', verbose=True, freeze_bn=True,mixed_precision=False,accumulation_steps=1,scheduler=None,):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            device=device,
            verbose=verbose,
        )
        
        self.freeze_bn=freeze_bn
        self.mixed_precision=mixed_precision
        self.accumulation_steps=accumulation_steps
        self.scheduler=scheduler
        self.optimizer = optimizer
        self.optimizer.zero_grad()
        self.steps = 0
        
        if self.mixed_precision:
            self.model, self.optimizer = amp.initialize(self.model,self.optimizer,opt_level="O1",verbosity=0)
            print('Mixed Precision ON')
            #self.freeze_bn = False
            #print('Unfrozen BN because of mixed precission training')
            
            
        if self.scheduler == 'one_cycle':
            self.scheduler = OneCycleScheduler(self.optimizer, num_steps=139*10, max_lr=[self.optimizer.param_groups[i]['lr'] for i in range(len(self.optimizer.param_groups))])


    def on_epoch_start(self):
        self.model.train()
        if self.freeze_bn:
            try:
                self.model.encoder.apply(fix_bn)
                print('Frozen BN')
            except:
                self.model.module.encoder.apply(fix_bn)
                print('Frozen BN')

    def batch_update(self, x, y):
        self.steps+=1
        prediction = self.model.forward(x)
        loss = self.loss(prediction, y)
        if self.mixed_precision:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        if (self.steps%self.accumulation_steps)==0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.scheduler == 'one_cycle':
                self.scheduler.step()
        return loss, prediction


class ValidEpoch(Epoch):

    def __init__(self, model, loss, metrics, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='valid',
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y)
        return loss, prediction
