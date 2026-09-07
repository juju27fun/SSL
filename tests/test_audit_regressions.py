import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from smi_ssl.cli import train_arrays
from smi_ssl.configuration import load_config
from smi_ssl.checkpoints import load_checkpoint
from smi_ssl.data import EventArrays,load_arrays,make_demo
from smi_ssl.monitoring import fixed_class_proportional_monitor_indices
from smi_ssl.training import evaluate_reconstruction


def test_monitor_selection_is_fixed_by_id_not_input_order(tmp_path):
    path=tmp_path/'events.npz';make_demo(path);data=load_arrays(path)
    train=EventArrays(data,'train')
    indices,meta=fixed_class_proportional_monitor_indices(train,max_samples=4,seed=42,split_tag='train')
    reference=json.loads((Path(__file__).parent/'fixtures/monitor_source.json').read_text())
    assert indices==reference['indices']
    assert meta==reference['metadata']
    ids=set(data['ids'][train.indices[indices]])
    train.indices=train.indices[::-1]
    reversed_indices,reversed_meta=fixed_class_proportional_monitor_indices(train,max_samples=4,seed=42,split_tag='train')
    assert ids==set(data['ids'][train.indices[reversed_indices]])
    assert meta==reversed_meta
    assert meta['class_counts']=={'2um':1,'4um':1,'10um':2}


@pytest.mark.parametrize('policy',['P25','CYCLIC25'])
def test_declared_evaluation_is_executed_and_matched_monitor_is_written(tmp_path,policy):
    torch.set_num_threads(1)
    path=tmp_path/'events.npz';make_demo(path)
    config=load_config();config['masking']['evaluation_policy']=policy
    config['training']['matched_monitoring']['samples_per_split']=3
    cp=tmp_path/'config.json';cp.write_text(json.dumps(config))
    run=tmp_path/'run';result=train_arrays(path,run,cp,'smoke','cpu')
    model,_=load_checkpoint(run/'checkpoints/latest.pt')
    data=load_arrays(path)
    for split,key in [('val','simulation_validation'),('real_val','real_validation')]:
        loader=DataLoader(EventArrays(data,split),batch_size=2)
        expected,_=evaluate_reconstruction(model,loader,config,torch.device('cpu'),mask_seed=42,evaluation_policy=policy)
        assert result[key]['model']['masked_mse']==pytest.approx(expected['model']['masked_mse'],rel=1e-5)
        assert result[key]['model']['masked_points']==expected['model']['masked_points']
    assert result['evaluation_mask_policy']==policy
    assert result['monitoring']['evaluation_policy']=='CYCLIC25'
    history=json.loads((run/'history.json').read_text())
    assert [r['epoch'] for r in history]==[0,1]
    for row in history:
        monitor=row['matched_monitor']
        assert monitor['evaluation_policy']=='CYCLIC25'
        assert monitor['train_eval']['model']['masked_points']>3*1024
        assert monitor['validation']['model']['masked_points']>3*1024


@pytest.mark.parametrize('protocol',['held-out','transductive'])
def test_unused_real_monitor_labels_do_not_block_clustering(tmp_path,protocol):
    rng=np.random.default_rng(42)
    labels=np.tile(np.arange(3),6);splits=np.repeat(['train','val','real_val'],6)
    labels[splits=='real_val']=-1
    if protocol=='transductive': labels[splits=='train']=-1
    source=tmp_path/'embeddings.npz'
    np.savez(source,embeddings=rng.normal(size=(18,6)),labels=labels,split=splits,
             ids=np.array([str(i) for i in range(18)]),groups=np.array([str(i) for i in range(18)]))
    output=tmp_path/'clusters.json'
    subprocess.run([str(Path(sys.executable).parent/'smi-ssl'),'cluster','--embeddings',str(source),'--output',str(output),'--protocol',protocol],check=True,capture_output=True,text=True)
    assert json.loads(output.read_text())['protocol']==protocol


def test_invalid_evaluation_policy_fails_before_training(tmp_path):
    config=load_config();config['masking']['evaluation_policy']='unknown'
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    with pytest.raises(ValueError,match='evaluation'):load_config(path)
