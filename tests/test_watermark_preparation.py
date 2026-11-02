import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from aquapilot.data.audit import audit_watermark
from aquapilot.watermark_grid import BANKS, GRID_NAME, subcarrier_grid
from scripts.prepare_watermark import WATERMARK_CHANNEL_CODES, convert_h_to_csi, main as single_fold
from scripts.prepare_watermark_loeo import main as four_folds


@pytest.fixture
def channels(tmp_path):
    root = tmp_path / 'input' / 'channels'
    for i, code in enumerate(WATERMARK_CHANNEL_CODES):
        folder = root / code / 'mat'
        folder.mkdir(parents=True)
        rng = np.random.default_rng(i)
        h = rng.normal(size=(18, 128)) + 1j * rng.normal(size=(18, 128))
        _,band,slow,fast = BANKS[code]
        savemat(folder / f'{code}_001.mat', {'h': h, 'fc': sum(band)/2, 'fs_t':slow, 'fs_tau':fast})
    return root


def convert(raw, output):
    four_folds(['--input', str(raw), '--output-root', str(output), '--n-subcarriers', '16'])


def test_published_folds_match_loader_and_frequency_grid(channels, tmp_path):
    from scipy.io import loadmat

    out = tmp_path / 'prepared'
    out.mkdir()  # A precreated empty destination is valid.
    convert(channels, out)
    status = audit_watermark(out, n_subcarriers=16, expected_counts={'NOF':1, 'NCS':1, 'BCH':1, 'KAU':2})
    assert (status['num_unique_records'], status['total_prepared_files']) == (5, 20)
    manifest = json.loads((out/'folds.json').read_text())
    for fold, layout in manifest['folds'].items():
        assert Path(layout['root']) == out/fold
        index = json.loads((out/fold/'index.json').read_text())
        assert index['format'] == 'AquaPilot prepared CSI'
        for record in index['records']:
            h = loadmat(record['source'])['h']
            _,band,_,fast = BANKS[record['channel_code']]
            hz = sum(band)/2 + (np.arange(16)-8)*(.8*(band[1]-band[0])/16)
            np.testing.assert_allclose(hz,record['subcarrier_hz'])
            positions=(hz-sum(band)/2)/fast
            fourier=np.exp(-2j*np.pi*np.arange(h.shape[1])[:,None]*positions)
            expected = (h@fourier).astype(np.complex64)
            with np.load(out/fold/record['prepared']) as z:
                np.testing.assert_allclose(z['csi_real']+1j*z['csi_imag'],expected,rtol=1e-6,atol=2e-6)
                assert Path(z['source'].item()).is_relative_to(channels)


@pytest.mark.parametrize('entry,flag', [(single_fold,'--output'), (four_folds,'--output-root')])
def test_malformed_selected_mat_does_not_leave_partial_data(channels,tmp_path,entry,flag):
    path = channels/'NOF1/mat/NOF1_001.mat'
    savemat(path, {'unrelated': np.ones((18,48))})
    out = tmp_path/'prepared'
    with pytest.raises(ValueError,match='NOF1_001.mat.*missing channel impulse response'):
        entry(['--input',str(channels),flag,str(out),'--n-subcarriers','16'])
    assert not out.exists() and not list(tmp_path.glob('.watermark-*'))


def test_duplicate_filename_after_normalization_is_rejected(channels,tmp_path):
    folder=channels/'NCS1/mat'
    for filename in ('NCS1_a b.mat','NCS1_a+b.mat'):
        (folder/filename).write_bytes((folder/'NCS1_001.mat').read_bytes())
    with pytest.raises(ValueError,match='same prepared record ID'):
        convert(channels,tmp_path/'prepared')
    assert not (tmp_path/'prepared').exists()


def test_missing_kau_receiver_does_not_publish_incomplete_four_folds(channels,tmp_path):
    (channels/'KAU2/mat/KAU2_001.mat').unlink()
    with pytest.raises(ValueError,match='Missing selected Watermark channels: KAU2'):
        convert(channels,tmp_path/'prepared')
    assert not (tmp_path/'prepared').exists()


def test_output_never_replaces_raw_or_existing_files(channels,tmp_path):
    out=tmp_path/'prepared'; out.mkdir(); prior=out/'existing'; prior.write_bytes(b'unmodified')
    with pytest.raises(FileExistsError,match='new or empty'): convert(channels,out)
    assert prior.read_bytes()==b'unmodified'
    for invalid in (channels.parent, channels, channels/'derived'):
        with pytest.raises(ValueError,match='separate directories'): convert(channels,invalid)
    assert len(list(channels.rglob('*.mat')))==5 and not (channels/'derived').exists()


def test_more_subcarriers_than_fft_bins_is_not_silent_duplication():
    with pytest.raises(ValueError,match='distinct frequency bins'):
        convert_h_to_csi(np.ones((12,8)), 16, .8, 0)


def test_long_sounding_axis_is_not_chosen_from_shorter_dimension():
    from scripts.verify_watermark_axes import verify_delay_transform
    frames, taps = 1024, 510  # NCS1: the slow-time dimension is longer than delay.
    h = np.zeros((frames,taps), dtype=np.complex64)
    t=np.arange(frames); gains=np.exp(2j*np.pi*t/107)
    h[:,5]=gains; h[:,11]=.25*gains.conj()
    kw=dict(channel_code='NCS1',fc=14000,fs_tau=16000); vkw=dict(probe_band_hz=(10000,18000),fc=14000,fs_tau=16000)
    got=convert_h_to_csi(h,128,.8,None,**kw)
    freq = 50*(np.arange(128)-64)  # 50 Hz spacing, from −3200 to +3150 Hz baseband
    target=h[:,5,None]*np.exp(-2j*np.pi*freq*5/16000) + h[:,11,None]*np.exp(-2j*np.pi*freq*11/16000)
    assert got.shape == (1024,128)
    np.testing.assert_allclose(got,target,rtol=3e-6,atol=3e-6)
    assert verify_delay_transform(h,got,.8,**vkw) < 3e-6
    with pytest.raises(ValueError,match='slow-time snapshots'):
        verify_delay_transform(h,convert_h_to_csi(h,128,.8,1,**kw),.8,**vkw)
    with pytest.raises(ValueError,match='delay-column DFT'):
        verify_delay_transform(h,got[:,::-1],.8,**vkw)


def test_mat_v73_returns_matlab_subscripts_for_compound_complex(tmp_path):
    import h5py
    from scripts.prepare_watermark import load_mat
    h = np.arange(9*5,dtype=float).reshape(9,5) * (1+2j)
    raw = tmp_path/'NCS1_001.mat'; fields=np.empty(h.T.shape,dtype=[('real','f8'),('imag','f8')]);fields['real']=h.T.real; fields['imag']=h.T.imag
    with h5py.File(raw,'w') as hf:
        ds=hf.create_dataset('h',data=fields);ds.attrs['MATLAB_class']=b'double'
    np.testing.assert_array_equal(load_mat(raw)['h'],h)
    with h5py.File(raw,'a') as hf: del hf['h'].attrs['MATLAB_class']
    with pytest.raises(ValueError,match='axis convention is unknown'):load_mat(raw)


def test_standard_audit_rejects_uncertified_axis_and_transposition():
    from aquapilot.data.audit import validate_standard_axes
    path=Path('NCS1_NCS1_001.npz'); old={'shape':[510,128],'frames':510}
    with pytest.raises(ValueError,match='Rebuild from the original MAT'):
        validate_standard_axes(old,{},'NCS1',path)
    padded,bins,hz = subcarrier_grid('NCS1',510,128,.8,14000,16000)
    attr={'matlab_h_shape':[1024,510],'slow_time_axis':1,'fft_delay_axis':0,'frequency_grid':GRID_NAME,
          'padded_dft_length':padded,'probe_band_hz':[10000,18000],'nominal_probe_bandwidth_hz':8000,'shifted_dft_bin_indices':bins.tolist(),
          'subcarrier_hz':hz.tolist(),'fc':14000,'fs_tau':16000,'fs_t':16000/510}
    with pytest.raises(ValueError,match='FFT must use the delay columns'):
        validate_standard_axes(dict(old,**attr),attr,'NCS1',path)


@pytest.mark.parametrize('code', sorted(BANKS))
def test_selected_tones_are_unique_and_inside_the_physical_probe_band(code):
    (rows, taps), band, _, rate = BANKS[code]
    fc = sum(band)/2; padded,indices,hz = subcarrier_grid(code,taps,128,.8,fc,rate)
    assert len(np.unique(indices)) == 128 and np.all(np.diff(hz)>0)
    halfwidth=.4*(band[1]-band[0]); assert hz[0]>=fc-halfwidth and hz[-1]<=fc+halfwidth
    spacing = .8*(band[1]-band[0])/128
    np.testing.assert_allclose(hz,fc+np.arange(-64,64)*spacing,rtol=0,atol=1e-10)
    np.testing.assert_allclose(hz,fc+(indices-padded//2)*rate/padded,rtol=0,atol=1e-10)
    # The earlier central-80%-of-full-delay-sampling grid extends beyond the probe in every bank.
    old_width=round(.8*taps);old_low=(taps-old_width)//2
    assert fc+(old_low-taps//2)*rate/taps < band[0]


def test_verifier_receipt_matches_ordered_fit_inputs(tmp_path, monkeypatch):
    from scripts import verify_watermark_axes as verifier
    from aquapilot.provenance import prepared_fingerprint
    raw=tmp_path/'raw'; prepared=tmp_path/'prepared'
    small={'BCH':1,'NOF':1,'NCS':1,'KAU':2}
    for i, (code,(shape,band,slow,rate)) in enumerate(BANKS.items(),1):
        folder=raw/code/'mat'; folder.mkdir(parents=True)
        h=np.zeros(shape,np.complex64); values=np.exp(2j*np.pi*np.arange(shape[0])/137)
        h[:,i]=values; h[:,2*i]=.1*values.conj()
        savemat(folder/f'{code}_001.mat',dict(h=h,fc=sum(band)/2,fs_t=slow,fs_tau=rate),do_compression=True)
    four_folds(['--input',str(raw),'--output-root',str(prepared)])
    # Use one constructed record per bank only in this integration test. The
    # production entry point always checks the full 156-record inventory.
    monkeypatch.setattr(verifier,'audit_watermark',lambda root:audit_watermark(root,expected_counts=small))
    receipt=tmp_path/'results/axes.json'
    verifier.main(['--raw-root',str(raw),'--prepared-root',str(prepared),'--output',str(receipt)])
    r=json.loads(receipt.read_text()); assert (r['num_unique_records'],r['total_prepared_files'])==(5,20)
    assert r['frequency_grid']==GRID_NAME and r['method']=='independent_delay_axis_fft'
    for group in small:
        assert r['ordered_split_fingerprints'][group]==prepared_fingerprint(prepared/('fold_test_'+group),('train','val','test'))['split_fingerprints']
    assert max(a['max_error_over_peak'] for a in r['environments']) < 4*np.finfo(np.float32).eps
    with pytest.raises(SystemExit): verifier.main(['--raw-root',str(raw),'--prepared-root',str(prepared),'--output',str(receipt)])
    # A training rerun must use the very files checked against the MAT matrices.
    from scripts import run_watermark_loeo as runner
    args = runner.parser().parse_args(["--prepare", "--raw-root", str(raw), "--prepared-root", str(prepared), "--axes", str(receipt)])
    receipt_bytes = receipt.read_bytes()
    runner.prepare_inputs(args)
    assert receipt.read_bytes() == receipt_bytes
    assert runner.match_receipt(args, r) == receipt
    sample = next((prepared/'fold_test_NCS/test').glob('*.npz'))
    with sample.open('ab') as handle:
        handle.write(b'changed archive')
    with pytest.raises(ValueError, match='inputs changed in fold_test_NCS'):
        runner.match_receipt(args, r)


def test_preflight_detects_tampered_physical_tone_indices():
    from aquapilot.data.audit import validate_standard_axes
    code='BCH1'; hshape,band,slow,rate=BANKS[code];fc=sum(band)/2
    padded,b,hz=subcarrier_grid(code,hshape[1],128,.8,fc,rate)
    row=dict(matlab_h_shape=list(hshape),slow_time_axis=0,fft_delay_axis=1,frequency_grid=GRID_NAME,
             padded_dft_length=padded,probe_band_hz=list(band),nominal_probe_bandwidth_hz=band[1]-band[0],
             shifted_dft_bin_indices=b.tolist(),subcarrier_hz=hz.tolist(),fc=fc,fs_tau=rate,fs_t=slow)
    rec=dict(row,frames=hshape[0],shape=[hshape[0],128])
    validate_standard_axes(rec,row,code,Path('BCH1_001.npz'))
    # Matching NPZ and JSON claims do not override the documented physical grid.
    for key, value in [('shifted_dft_bin_indices',(b+1).tolist()),('subcarrier_hz',(hz+1).tolist())]:
        with pytest.raises(ValueError,match='frequency samples'):
            validate_standard_axes(dict(rec,**{key:value}),dict(row,**{key:value}),code,Path('BCH1_001.npz'))
