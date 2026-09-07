import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from smi_ssl.configuration import load_config, make_model
from smi_ssl.data import make_demo, load_arrays, EventArrays
from smi_ssl.checkpoints import load_checkpoint
from smi_ssl.cli import train_arrays
from smi_ssl.training import build_training_mask_batch

torch.set_num_threads(1)


def test_source_groups_cannot_cross_splits(tmp_path):
    path=tmp_path/'demo.npz';make_demo(path)
    payload=load_arrays(path);payload['groups'][6]=payload['groups'][0]
    bad=tmp_path/'bad.npz';np.savez(bad,**payload)
    with pytest.raises(ValueError,match='crosses splits'):
        load_arrays(bad)


def test_per_window_normalization_does_not_use_other_rows(tmp_path):
    path=tmp_path/'demo.npz';make_demo(path)
    data=load_arrays(path);first=EventArrays(data,'train')[0]['signal'].numpy().copy()
    data['signals'][1] *= 1000
    np.testing.assert_array_equal(first,EventArrays(data,'train')[0]['signal'].numpy())
    assert abs(float(first.mean()))<1e-5
    assert float(first.std())==pytest.approx(1.0,abs=1e-5)


@pytest.mark.parametrize('policy',['P25','CYCLIC25'])
def test_masks_define_hidden_loss_targets(tmp_path,policy):
    path=tmp_path/'demo.npz';make_demo(path)
    data=EventArrays(load_arrays(path),'train')
    batch=next(iter(torch.utils.data.DataLoader(data,batch_size=2)))
    config=load_config();config['masking']['training_policy']=policy
    mask=build_training_mask_batch(batch,config,seed=42,cycle_step=0)
    assert mask.dtype==torch.bool and mask.shape==(2,4096)
    assert (mask.sum(dim=1)==1024).all()
    model=make_model(config).eval()
    changed=batch['signal'].clone();changed[mask.unsqueeze(1)]=9999
    with torch.no_grad():
        torch.testing.assert_close(model(batch['signal'],time_mask=mask),model(changed,time_mask=mask))


def test_one_epoch_checkpoint_roundtrip_and_outside_cwd(tmp_path):
    path=tmp_path/'demo.npz';make_demo(path)
    run=tmp_path/'run'
    result=train_arrays(path,run,None,'smoke','cpu')
    assert result['selected_epoch']==1 and result['checkpoint_selection']=='fixed_final'
    model,payload=load_checkpoint(run/'checkpoints/latest.pt')
    assert payload['epoch']==1 and not model.training
    with torch.no_grad():
        assert model(torch.zeros(1,1,4096)).shape==(1,1,4096)
    out=tmp_path/'config.json'
    command=[str(Path(sys.executable).parent/'smi-ssl'),'config','--output',str(out)]
    subprocess.run(command,cwd=tmp_path,check=True,capture_output=True,text=True)
    assert json.loads(out.read_text())['model']['architecture']=='unet1d_reconstructor'


def test_prepare_manifest_crops_and_rejects_leakage_without_output(tmp_path):
    import csv
    from smi_ssl.data import prepare_manifest
    rows=[]
    for i,split in enumerate(('train','val','real_val')):
        np.save(tmp_path/f'{i}.npy', np.arange(5000,dtype=np.float32)+i)
        rows.append(dict(path=f'{i}.npy',id=str(i),split=split,center_sample=2500,
                         start_sample=2200,end_sample=2800,group=str(i),label=i))
    manifest=tmp_path/'events.csv'
    def write():
        with manifest.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    write();output=tmp_path/'prepared.npz';prepare_manifest(manifest,tmp_path,output)
    data=load_arrays(output)
    np.testing.assert_array_equal(data['signals'][0],np.arange(452,4548))
    assert data['event_masks'][0].sum()==600
    rows[1]['group']='0';write();bad=tmp_path/'invalid.npz'
    with pytest.raises(ValueError,match='crosses splits'):
        prepare_manifest(manifest,tmp_path,bad)
    assert not bad.exists()
