import torch
import os
import torch.nn.functional as Func
import numpy as np
import math
from torch.utils.data import DataLoader
import torch.nn as nn
from model import UNet
from util import tf2pytorch
#from librosa.output import write_wav
#from torchaudio.functional import istft
import soundfile as sf
import torchaudio
import datetime
import ffmpeg
import scipy
from scipy.signal.windows import hann
from librosa.core import stft, istft
import datetime

'''
Usage:
    from separator import Separator
    sep = Separator(params)
    sep.separate(input_wav_dir, output_dir(optional))
    sep.batch_separate(input_wav_files, output_dir(optional))
'''

params = {
    'sample_rate': 44100,
    'frame_length': 4096,
    'frame_step': 1024,
    'T': 512,
    'F': 1024,
    'num_instruments': ['vocal', 'instrumental'],
    #'resume': ['./final_model/net_vocal.pth', './final_model/net_instrumental.pth'],
    'resume': None,
    'output_dir': './output',
    'checkpoint_path': './2stems/model'
}

device = torch.device('cuda')

class Separator(object):
    def __init__(self, params=params):
        self.num_instruments = params['num_instruments']
        self.model_list = nn.ModuleList()
        if params['resume']:
            for i in range(len(params['num_instruments'])):
                print('Loading model for instrumment {}'.format(i))
                checkpoint = torch.load(params['resume'][i])
                net = UNet()
                net.load_state_dict(checkpoint['state_dict'])
                net.to(device)
                self.model_list.append(net)
        else:
            ckpts = tf2pytorch(params['checkpoint_path'], params['num_instruments'])
            for i in range(len(self.num_instruments)):
                print('Loading model for instrumment {}'.format(i))
                net = UNet().to(device)
                ckpt = ckpts[i]
                net = self._load_ckpt(net, ckpt)
                net.eval()
                self.model_list.append(net)

        self.T = params['T']
        self.F = params['F']
        self.frame_length = params['frame_length']
        self.frame_step = params['frame_step']
        self.samplerate = params['sample_rate']


    def _load_ckpt(self, model, ckpt):
        state_dict = model.state_dict()
        for k, v in ckpt.items():
            if k in state_dict:
                target_shape = state_dict[k].shape
                assert target_shape == v.shape
                state_dict.update({k: torch.from_numpy(v)})
            else:
                print('Ignore ', k)

        model.load_state_dict(state_dict)
        return model

 
    def _load_audio(
            self, path, offset=None, duration=None,
            sample_rate=None, dtype=np.float32):
        """ Loads the audio file denoted by the given path
        and returns it data as a waveform.

        :param path: Path of the audio file to load data from.
        :param offset: (Optional) Start offset to load from in seconds.
        :param duration: (Optional) Duration to load in seconds.
        :param sample_rate: (Optional) Sample rate to load audio with.
        :param dtype: (Optional) Numpy data type to use, default to float32.
        :returns: Loaded data a (waveform, sample_rate) tuple.
        :raise SpleeterError: If any error occurs while loading audio.
        """
        if not isinstance(path, str):
            path = path.decode()
        
        probe = ffmpeg.probe(path)

        metadata = next(
            stream
            for stream in probe['streams']
            if stream['codec_type'] == 'audio')
        n_channels = metadata['channels']
        if sample_rate is None:
            sample_rate = metadata['sample_rate']
        output_kwargs = {'format': 'f32le', 'ar': sample_rate}
        #if duration is not None:
        #    output_kwargs['t'] = _to_ffmpeg_time(duration)
        #if offset is not None:
        #    output_kwargs['ss'] = _to_ffmpeg_time(offset)
        process = (
            ffmpeg
            .input(path)
            .output('pipe:', **output_kwargs)
            .run_async(pipe_stdout=True, pipe_stderr=True))
        buffer, _ = process.communicate()
        waveform = np.frombuffer(buffer, dtype='<f4').reshape(-1, n_channels)
        if not waveform.dtype == np.dtype(dtype):
            waveform = waveform.astype(dtype)
        return waveform, sample_rate
    
    def _to_ffmpeg_codec(codec):
        ffmpeg_codecs = {
            'm4a': 'aac',
            'ogg': 'libvorbis',
            'wma': 'wmav2',
        }
        return ffmpeg_codecs.get(codec) or codec

    def _save_to_file(
            self, path, data, sample_rate,
            codec=None, bitrate=None):
        """ Write waveform data to the file denoted by the given path
        using FFMPEG process.

        :param path: Path of the audio file to save data in.
        :param data: Waveform data to write.
        :param sample_rate: Sample rate to write file in.
        :param codec: (Optional) Writing codec to use.
        :param bitrate: (Optional) Bitrate of the written audio file.
        :raise IOError: If any error occurs while using FFMPEG to write data.
        """
        directory = os.path.dirname(path)
        #get_logger().debug('Writing file %s', path)
        input_kwargs = {'ar': sample_rate, 'ac': data.shape[1]}
        output_kwargs = {'ar': sample_rate, 'strict': '-2'}
        if bitrate:
            output_kwargs['audio_bitrate'] = bitrate
        if codec is not None and codec != 'wav':
            output_kwargs['codec'] = _to_ffmpeg_codec(codec)
        
        process = (
            ffmpeg
            .input('pipe:', format='f32le', **input_kwargs)
            .output(path, **output_kwargs)
            .overwrite_output()
            .run_async(pipe_stdin=True, pipe_stderr=True, quiet=True))
        
        process.stdin.write(data.astype('<f4').tobytes())
        process.stdin.close()
        process.wait()

    def _stft(self, data, inverse=False, length=None):
        """
        Single entrypoint for both stft and istft. This computes stft and istft with librosa on stereo data. The two
        channels are processed separately and are concatenated together in the result. The expected input formats are:
        (n_samples, 2) for stft and (T, F, 2) for istft.
        :param data: np.array with either the waveform or the complex spectrogram depending on the parameter inverse
        :param inverse: should a stft or an istft be computed.
        :return: Stereo data as numpy array for the transform. The channels are stored in the last dimension
        """
        assert not (inverse and length is None)
        data = np.asfortranarray(data)
        N = self.frame_length
        H = self.frame_step
        win = hann(N, sym=False)
        fstft = istft if inverse else stft
        win_len_arg = {"win_length": None, "length": length} if inverse else {"n_fft": N}
        n_channels = data.shape[-1]
        out = []
        for c in range(n_channels):
            d = data[:, :, c].T if inverse else data[:, c]
            s = fstft(d, hop_length=H, window=win, center=False, **win_len_arg)
            s = np.expand_dims(s.T, 2-inverse)
            out.append(s)
        if len(out) == 1:
            return out[0]
        return np.concatenate(out, axis=2-inverse)



    def _pad_and_partition(self, tensor, T):
        old_size = tensor.size(3)
        new_size = math.ceil(old_size/T) * T
        tensor = Func.pad(tensor, [0, new_size - old_size])
        [b, c, t, f] = tensor.shape
        split = new_size // T
        
        return torch.cat(torch.split(tensor, T, dim=3), dim=0)

    

    def separate(self, input_wav_list, output_dir='./output'):
        B_length_list = []
        L_length_list = []
        wav_name_list = []
        stft_list = []
        stft_mag_list = []

        for input_wav in input_wav_list:
            wav_name = input_wav.split('/')[-1].split('.')[0]
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            source_audio, samplerate = self._load_audio(input_wav)  # Length * 2    
            source_audio = torch.from_numpy(np.array(source_audio)).T # 2 * Length
    

            if int(samplerate) != 44100:
                resample = torchaudio.transforms.Resample(int(samplerate), 44100)
                source_audio = resample(source_audio)
                samplerate = 44100
        

            if source_audio.shape[0] == 1: 
                source_audio = torch.cat((source_audio, source_audio), dim=0)
            elif source_audio.shape[0] > 2:
                source_audio = source_audio[:2, :]
        
            stft = self._stft(source_audio.T) # L * F * 2
            stft = stft[:, : self.F, :]
        
            stft_mag = abs(stft) # L * F * 2
            stft_mag = torch.from_numpy(stft_mag)
            stft_mag = stft_mag.unsqueeze(0).permute([0, 3, 2, 1]) # 1 * 2 * F * L

            L = stft.shape[0]

            stft_mag = self._pad_and_partition(stft_mag, self.T) # [(L + T) / T] * 2 * F * T
            stft_mag = stft_mag.transpose(2, 3)
            # stft_mag : B * 2 * T * F

            B = stft_mag.shape[0] 

            wav_name_list.append(wav_name)
            B_length_list.append(B)
            L_length_list.append(L)
            stft_list.append(stft)
            stft_mag_list.append(stft_mag)

        stft_mag_all = torch.cat(stft_mag_list, dim=0)
        stft_mag_all = stft_mag_all.to(device)
        
        masks = []

        for model in self.model_list:
            mask = model(stft_mag_all, output_mask_logit=True)
            masks.append(mask)

        
        mask_sum = sum([m ** 2 for m in masks])
        mask_sum += 1e-10
        masks_norm = []

        for i in range(len(self.num_instruments)):
            mask = masks[i]
            mask = (mask ** 2 + 1e-10/2) / (mask_sum)
            mask = mask.transpose(2, 3)  # B x 2 X F x T
            masks_norm.append(mask)
        
        B_start_index = 0

        for i in range(len(wav_name_list)):
            wav_name = wav_name_list[i]
            B = B_length_list[i]
            L = L_length_list[i]
            stft = stft_list[i]
            
            for j in range(len(self.num_instruments)):
                mask = masks_norm[j]
                mask = mask[B_start_index: B_start_index + L, :, :, :]

                mask = torch.cat(torch.split(mask, 1, dim=0), dim=3)
                mask = mask.squeeze(0)[:,:,:L] # 2 x F x L
                mask = mask.permute([2, 1, 0])
                mask = mask.cpu()
                mask = mask.detach().numpy()

                stft_masked = stft * mask              
                stft_masked = np.pad(stft_masked, ((0,0),(0,1025),(0,0)), 'constant')
            
                wav_masked = self._stft(stft_masked, inverse=True, length=source_audio.shape[1])
                save_path = os.path.join(output_dir, (wav_name + '-' + self.num_instruments[j] + '.wav'))
             
                self._save_to_file(save_path, wav_masked, samplerate, 'wav', '128k')
            B_start_index += B
            print('Audio {} separated'.format(wav_name))
