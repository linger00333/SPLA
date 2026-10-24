"""Run the formal 10-epoch pilot with matched pre-training validation evidence."""
from pathlib import Path
import json,os,random,runpy,sys,time
import numpy as np
import torch
KIT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(KIT),str(KIT.parent/'framework')]
from ultralytics.models.yolo.detect import DetectionTrainer

original_init=DetectionTrainer.__init__
def audit_before_training(trainer):
    params=list(trainer.model.named_parameters())
    frozen=[n for n,p in params if not p.requires_grad]
    backbones=[(n,p) for n,p in params if n.startswith('model.') and int(n.split('.')[1])<=20]
    if trainer.args.freeze==21:assert all(not p.requires_grad for n,p in backbones)
    initial={'args':vars(trainer.args),'frozen_parameter_names':frozen,'frozen_parameters':sum(p.numel() for n,p in params if not p.requires_grad),'trainable_parameters':sum(p.numel() for n,p in params if p.requires_grad),'all_backbone_parameters_frozen':all(not p.requires_grad for n,p in backbones),'batchnorm_statistics_not_explicitly_frozen':True,'lr_by_epoch':[trainer.args.lr0*trainer.lf(i) for i in range(trainer.epochs)],'prior_source':'no SPLA layers in A1c'}
    (trainer.save_dir/'runtime_audit.json').write_text(json.dumps(initial,indent=2,default=str))
    assert all(abs(lr-0.0001)<1e-12 for lr in initial['lr_by_epoch'])
    rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state(trainer.device))
    gen=getattr(trainer.test_loader,'generator',None)
    gs=gen.get_state() if gen is not None else None
    missing=object()
    previous_loss,previous_epoch=getattr(trainer,'loss_items',missing),getattr(trainer,'epoch',missing)
    trainer.loss_items=torch.zeros(3,device=trainer.device)
    trainer.epoch=-1
    started=time.time()
    try:metrics=trainer.validator(trainer=trainer)
    finally:
        for key,value in [('loss_items',previous_loss),('epoch',previous_epoch)]:
            if value is missing:delattr(trainer,key)
            else:setattr(trainer,key,value)
        random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2]);torch.cuda.set_rng_state(rng[3],trainer.device)
        if gs is not None:gen.set_state(gs)
    (trainer.save_dir/'pretrain_validation.json').write_text(json.dumps({'metrics':metrics,'elapsed_seconds':time.time()-started,'same_training_validator':True,'random_states_restored':True},indent=2,default=float))
def init(self,*args,**kwargs):
    original_init(self,*args,**kwargs)
    self.add_callback('on_pretrain_routine_end',audit_before_training)
DetectionTrainer.__init__=init
sys.argv=[str(KIT/'tools/train_spla.py'),*sys.argv[1:]]
runpy.run_path(str(KIT/'tools/train_spla.py'),run_name='__main__')
