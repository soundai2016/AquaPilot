"""Local numeric controls for the read-only diagnostic; no measured data are distributed."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat

from aquapilot.provenance import prepared_fingerprint
from scripts.check_watermark_axes import diagnose, FOLDS, SPLITS


class SourceComparison(unittest.TestCase):
    def test_distinguishes_swapped_axes_out_of_band_and_correct_on_identical_source(self):
        # A single delayed arrival whose complex amplitude varies in slow time.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = root/'channels/NCS1/mat'; raw.mkdir(parents=True)
            h = np.zeros((1024,510), dtype=np.complex64)
            amp = np.exp(.08j*np.arange(1024))
            h[:,17] = amp
            savemat(raw/'NCS1_001.mat',{'h':h,'fs_t':16000/510,'fs_tau':16000.,'fc':14000.})
            physical = amp[:,None]*np.exp((-2j*np.pi*17/16000)*(50*np.arange(-64,64)))
            def old(axis):
                # Old converter used an inclusive linspace over the middle of fftshift.
                L=axis.shape[1]; width=round(.8*L); left=(L-width)//2
                idx=np.linspace(left,left+width-1,128,dtype=int)
                return np.fft.fftshift(np.fft.fft(axis,axis=1),axes=1)[:,idx]
            prepared=root/'npz'
            cases=[physical, old(h), old(h.T), np.zeros_like(physical)]
            for fold,values in zip(FOLDS,cases):
                test=prepared/('fold_test_'+fold)/'test'; test.mkdir(parents=True)
                np.savez(test/'NCS1_NCS1_001.npz',csi=values.astype(np.complex64))
            result=diagnose(raw.parents[1],prepared,record='NCS1_001')
            self.assertEqual(result['counts'],{'in_band':1,'old_band':1,'exchanged_old_band':1,'other':1})
            self.assertEqual(result['compared_copies'],4)

    def test_raw_mat_not_required_to_match_ordered_training_fingerprints(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder); prep=base/'prepared'
            trained={}
            for fold in FOLDS:
                root=prep/('fold_test_'+fold)
                for name in SPLITS:
                    split=root/name; split.mkdir(parents=True)
                    np.savez(split/'trace.npz', csi=np.array([1j]))
                snapshot=prepared_fingerprint(root,SPLITS)
                trained[fold]={k:{'num_files':1,'ordered_input_fingerprint':h}
                               for k,h in snapshot['split_fingerprints'].items()}
            report=diagnose(base/'missing_raw',prep,trained_manifest=trained)
            self.assertEqual(report['status'],'unavailable')
            self.assertEqual(report['trained_input_binding']['matched_splits'],12)
            np.savez(prep/'fold_test_NCS/test/trace.npz', csi=np.array([2j]))
            report=diagnose(base/'missing_raw',prep,trained_manifest=trained)
            self.assertEqual(report['trained_input_binding']['status'],'mismatch')
            self.assertEqual(report['trained_input_binding']['matched_splits'],11)
