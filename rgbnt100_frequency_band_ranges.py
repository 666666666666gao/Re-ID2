"""One internal FFT-mask parameter pair on the unchanged global AV42 framework."""
import torch

from axis_collaboration import FrequencyAxisExpert,ModalityAxisExpert
from rgbnt100_expert_seed_recipe import build_selected,PARAMETERS


def revision(low,high):
    return (f'RGBNT100 global AV42 with only spatial FFT cutoffs {low}/{high}. '
        'Existing three-band masks change for both M and F; original global readers, '
        'identity teacher42, fresh expert42/Adam, losses, routing, projections, gates, '
        'optimizer groups, data order, BN2/partial0/two backwards and descriptor5120 '
        'remain. No new learned parameters or failed local-grid reader. Ordinary '
        'receives identical cutoffs. Full8675train/1715query/8575gallery, extra50, '
        'earliest normal benchmarkmAP-best/all six same checkpoint. Known parameter '
        'tuning, not a new module, energy-balance claim or established retrieval gain.')


def build_ranges(low,high,args,cfg,classes,cameras):
    assert args.dataset=='RGBNT100' and args.seed==42 and args.freeze_identity_encoder==1
    assert (low,high) in ((.125,.25),(.25,.375))
    assert tuple(cfg.INPUT.SIZE_TRAIN)==tuple(cfg.INPUT.SIZE_TEST)==(128,256)
    assert tuple(cfg.MODEL.STRIDE_SIZE)==(16,16)
    model=build_selected('AV',args,cfg,classes,cameras)
    assert model.modality_expert.read.__func__ is ModalityAxisExpert.read
    assert model.frequency_expert.prepare.__func__ is FrequencyAxisExpert.prepare
    assert model.frequency_expert.read.__func__ is FrequencyAxisExpert.read
    before=set(model.state_dict());parameters={n:id(p) for n,p in model.named_parameters()}
    device=model.bands.masks.device
    fy=torch.fft.fftfreq(8,device=device)[:,None]
    fx=torch.fft.fftfreq(16,device=device)[None,:]
    radius=(fy.square()+fx.square()).sqrt()
    masks=torch.stack((radius<=low,(radius>low)&(radius<=high),radius>high))
    assert masks.shape==model.bands.masks.shape==(3,8,16)
    assert torch.equal(masks.sum(0),torch.ones((8,16),device=device,dtype=torch.long))
    iy=(-torch.arange(8,device=device))%8;ix=(-torch.arange(16,device=device))%16
    assert torch.equal(masks,masks[:,iy][:,:,ix])
    bins=masks.sum((1,2)).tolist()
    assert bins==([7,18,103] if (low,high)==(.125,.25) else [25,30,73])
    model.bands.masks.copy_(masks)
    args.band_low_cutoff,args.band_high_cutoff=low,high
    assert set(model.state_dict())==before and {n:id(p) for n,p in model.named_parameters()}==parameters
    assert (sum(p.numel() for p in model.parameters()),sum(p.numel() for p in model.parameters() if p.requires_grad))==PARAMETERS['AV']
    model.band_range_reconstruction_errors=[]
    if args.mode=='smoke':
        def reconstruction(module,inputs,output):
            error=(output.sum(2)-inputs[0].float()).abs().max()
            assert torch.isfinite(error) and error<1e-5
            model.band_range_reconstruction_errors.append(float(error.detach()))
        model.bands.register_forward_hook(reconstruction)
    model.anchor_record.update(method=revision(low,high),
        frequency_band_ranges=dict(low=low,high=high,bin_counts=bins,grid=[8,16],parameters_added=0,
            complete_disjoint_conjugate_symmetric=True,global_M_F_readers_unchanged=True),
        construction_scope='Original global AV42 fresh-expert construction; only existing persistent band-mask values changed')
    return model
