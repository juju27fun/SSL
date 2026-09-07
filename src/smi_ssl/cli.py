"""Commands for explicit input arrays and bounded CPU smoke runs."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .configuration import load_config
from .data import EventArrays, load_arrays, make_demo, prepare_manifest
from .training import train_bead_ssl
from .checkpoints import load_checkpoint
from .embedding import extract_embeddings
from .decimation import normalize_signal
from .clustering import prepare_latents, evaluate_partition, summarize_partitions


def jsonable(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [jsonable(v) for v in value]
    return value


def train_arrays(data: Path, output: Path, config_path: Path | None, profile: str, device: str):
    config=load_config(config_path)
    payload=load_arrays(data)
    p=config['training']['profiles'][profile]
    datasets=(EventArrays(payload,'train',p['max_simulation_train']),
              EventArrays(payload,'val',p['max_simulation_validation']),
              EventArrays(payload,'real_val'))
    return train_bead_ssl(config,simulation_root=None,real_root=None,output_dir=output,
                          profile_name=profile,device_name=device,prepared_datasets=datasets)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    demo=commands.add_parser('demo-data');demo.add_argument('--output',type=Path,required=True);demo.add_argument('--seed',type=int,default=42)
    prepare=commands.add_parser('prepare');prepare.add_argument('--manifest',type=Path,required=True);prepare.add_argument('--signal-root',type=Path,required=True);prepare.add_argument('--output',type=Path,required=True)
    train=commands.add_parser('train');train.add_argument('--data',type=Path,required=True);train.add_argument('--output',type=Path,required=True);train.add_argument('--config',type=Path);train.add_argument('--profile',choices=['smoke','full'],default='smoke');train.add_argument('--device',default='cpu')
    config=commands.add_parser('config');config.add_argument('--output',type=Path,required=True)
    encode=commands.add_parser('encode');encode.add_argument('--data',type=Path,required=True);encode.add_argument('--checkpoint',type=Path,required=True);encode.add_argument('--output',type=Path,required=True);encode.add_argument('--device',default='cpu');encode.add_argument('--batch-size',type=int,default=32)
    cluster=commands.add_parser('cluster');cluster.add_argument('--embeddings',type=Path,required=True);cluster.add_argument('--output',type=Path,required=True);cluster.add_argument('--protocol',choices=['held-out','transductive'],default='held-out');cluster.add_argument('--classes',type=int,default=3)
    args=parser.parse_args();torch.set_num_threads(1)
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.command=='demo-data': make_demo(args.output,args.seed)
    elif args.command=='prepare': prepare_manifest(args.manifest,args.signal_root,args.output)
    elif args.command=='config': args.output.write_text(json.dumps(load_config(),indent=2)+'\n')
    elif args.command=='train':
        result=train_arrays(args.data,args.output,args.config,args.profile,args.device)
        print(json.dumps({'selected_epoch':result['selected_epoch'],'output':str(args.output),'purpose':'development; a smoke profile is not a scientific reproduction'}))
    elif args.command=='encode':
        data=load_arrays(args.data);model,payload=load_checkpoint(args.checkpoint,args.device)
        values=np.stack([normalize_signal(x,mode=payload['config']['data']['normalization']) for x in data['signals']])
        embeddings=extract_embeddings(model,values,device=torch.device(args.device),batch_size=args.batch_size)
        np.savez_compressed(args.output,embeddings=embeddings,split=data['split'],ids=data['ids'],groups=data['groups'],labels=data['labels'])
    elif args.command=='cluster':
        with np.load(args.embeddings,allow_pickle=False) as data:
            features=data['embeddings'];labels=data['labels'];splits=data['split'];groups=data['groups'];ids=data['ids']
        if labels.ndim!=1 or len(labels)!=len(features) or set(labels)!=set(range(args.classes)):
            raise ValueError('Labels must cover integer class IDs 0..classes-1')
        train=splits=='train';val=splits=='val'
        if not train.any() or not val.any(): raise ValueError('Train and val partitions are required')
        if set(groups[train]) & set(groups[val]) or set(ids[train]) & set(ids[val]):
            raise ValueError('Train/validation group or ID overlap')
        fit=val if args.protocol=='transductive' else train
        prepared=prepare_latents(features[fit],features[val])
        parts=[evaluate_partition(prepared,labels[fit],labels[val],seed=seed,n_classes=args.classes) for seed in (41,42,43)]
        result={'protocol':args.protocol,'fit_population':'val' if args.protocol=='transductive' else 'train','scored_population':'val','claim_boundary':'same-population descriptive clustering' if args.protocol=='transductive' else 'held-out validation with train-fitted transforms and Hungarian mapping','summary':summarize_partitions(parts),'partitions':parts}
        args.output.write_text(json.dumps(jsonable(result),indent=2,allow_nan=False)+'\n')
    print(args.output)
