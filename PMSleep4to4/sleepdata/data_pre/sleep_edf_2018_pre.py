import os
import glob
import ntpath
import logging
import argparse

import pyedflib
import numpy as np

# Label values
W = 0
N1 = 1
N2 = 2
N3 = 3
REM = 4
MOVE = 5
UNK = 6

stage_dict = {
    "W": W,
    "N1": N1,
    "N2": N2,
    "N3": N3,
    "REM": REM,
    "MOVE": MOVE,
    "UNK": UNK
}

# Have to manually define based on the dataset
ann2label = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3, "Sleep stage 4": 3,  # Follow AASM Manual
    "Sleep stage R": 4,
    "Sleep stage ?": 6,
    "Movement time": 5
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="E:\sleepstage\sleepdata\sleep-edf-database-expanded-1.0.0\sleep-cassette",
                        help="File path to the Sleep-EDF dataset.")
    parser.add_argument("--output_dir", type=str, default="./sleep2018_npz",
                        help="Directory where to save outputs.")
    parser.add_argument("--select_ch", type=str, default="[EEG Fpz-Cz,EEG Pz-Oz,EOG horizontal]",
                        help="Name of the channel in the dataset.")
    parser.add_argument("--log_file", type=str, default="info_ch_extract.log",
                        help="Log file.")
    args = parser.parse_args()

    # Output dir
    args.output_dir = os.path.join(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    args.log_file = os.path.join(os.path.join(args.output_dir, args.log_file))
    prompts_ch = ['Resp oro-nasal', 'EMG submental', 'Temp rectal', 'Event marker']

    # 数据的原生采样频率，sampling_rate是使用的数据采样频率，prompt_rate是数据中事件数据的采样频率
    sampling_rate = 100.0
    prompt_rate = 1.0

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(args.log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Select channel
    select_ch = args.select_ch
    #将字符串转换为列表
    select_ch = select_ch.replace('[', '')
    select_ch = select_ch.replace(']', '')
    select_ch = select_ch.replace('\'', '')
    select_ch = select_ch.split(',')
    # Read raw and annotation from EDF files
    psg_fnames = glob.glob(os.path.join(args.data_dir, "*PSG.edf"))
    ann_fnames = glob.glob(os.path.join(args.data_dir, "*Hypnogram.edf"))
    psg_fnames.sort()
    ann_fnames.sort()
    psg_fnames = np.asarray(psg_fnames)
    ann_fnames = np.asarray(ann_fnames)

    for i in range(len(psg_fnames)):

        logger.info("Loading ...")
        logger.info("Signal file: {}".format(psg_fnames[i]))
        logger.info("Annotation file: {}".format(ann_fnames[i]))

        psg_f = pyedflib.EdfReader(psg_fnames[i])
        ann_f = pyedflib.EdfReader(ann_fnames[i])

        assert psg_f.getStartdatetime() == ann_f.getStartdatetime()
        start_datetime = psg_f.getStartdatetime()
        logger.info("Start datetime: {}".format(str(start_datetime)))

        file_duration = psg_f.getFileDuration()
        logger.info("File duration: {} sec".format(file_duration))
        epoch_duration = psg_f.datarecord_duration
        if psg_f.datarecord_duration == 60:  # Fix problems of SC4362F0-PSG.edf, SC4362FC-Hypnogram.edf
            epoch_duration = epoch_duration / 2
            logger.info("Epoch duration: {} sec (changed from 60 sec)".format(epoch_duration))
        else:
            logger.info("Epoch duration: {} sec".format(epoch_duration))

        # Extract signal from the selected channel
        ch_names = psg_f.getSignalLabels()
        ch_samples = psg_f.getNSamples()
        select_ch_idx = []
        for s in range(psg_f.signals_in_file):
            for sj in range(len(select_ch)):
                if ch_names[s] == select_ch[sj]:
                    select_ch_idx.append(s)
                    break
        if select_ch_idx == []:
            raise Exception("Channel not found.")

        # 剩余通道作为提示词通道
        prompt_index = []
        for p in range(psg_f.signals_in_file):
            for pj in range(len(prompts_ch)):
                if ch_names[p] == prompts_ch[pj]:
                    prompt_index.append(p)
                    break
        if len(prompt_index) != len(prompts_ch):
            raise Exception("Channel not found.")

        n_epoch_samples = int(epoch_duration * sampling_rate)


        for s in select_ch_idx:
            if s == select_ch_idx[0]:
                signals = psg_f.readSignal(s).reshape(-1, n_epoch_samples)
                #添加一个维度
                signals = signals[np.newaxis, :, :]
            else:
                signals_tmp = psg_f.readSignal(s).reshape(-1, n_epoch_samples)
                signals_tmp = signals_tmp[np.newaxis, :, :]
                signals = np.concatenate((signals, signals_tmp), axis=0)

        n_epoch_samples_prompt = int(epoch_duration * prompt_rate)
        for p in prompt_index:
            if p == prompt_index[0]:
                prompt = psg_f.readSignal(p).reshape(-1, n_epoch_samples_prompt)
                prompt = prompt[np.newaxis, :, :]
            else:
                prompt_tmp = psg_f.readSignal(p).reshape(-1, n_epoch_samples_prompt)
                prompt_tmp = prompt_tmp[np.newaxis, :, :]
                prompt = np.concatenate((prompt, prompt_tmp), axis=0)

        # Sanity check
        n_epochs = psg_f.datarecords_in_file
        if psg_f.datarecord_duration == 60:  # Fix problems of SC4362F0-PSG.edf, SC4362FC-Hypnogram.edf
            n_epochs = n_epochs * 2
        assert signals.shape[1] == n_epochs, f"signal: {signals.shape} != {n_epochs}"
        assert prompt.shape[1] == n_epochs, f"prompt: {prompt.shape} != {n_epochs}"
        # Generate labels from onset and duration annotation
        labels = []
        total_duration = 0
        ann_onsets, ann_durations, ann_stages = ann_f.readAnnotations()
        for a in range(len(ann_stages)):
            onset_sec = int(ann_onsets[a])
            duration_sec = int(ann_durations[a])
            ann_str = "".join(ann_stages[a])

            # Sanity check
            assert onset_sec == total_duration

            # Get label value
            label = ann2label[ann_str]

            # Compute # of epoch for this stage
            if duration_sec % epoch_duration != 0:
                logger.info(f"Something wrong: {duration_sec} {epoch_duration}")
                raise Exception(f"Something wrong: {duration_sec} {epoch_duration}")
            duration_epoch = int(duration_sec / epoch_duration)

            # Generate sleep stage labels
            label_epoch = np.ones(duration_epoch, dtype=np.int32) * label
            labels.append(label_epoch)

            total_duration += duration_sec

            logger.info("Include onset:{}, duration:{}, label:{} ({})".format(
                onset_sec, duration_sec, label, ann_str
            ))
        labels = np.hstack(labels)

        # Remove annotations that are longer than the recorded signals
        labels = labels[:signals.shape[1]]

        # Get epochs and their corresponding labels
        x = signals.astype(np.float32)
        y = labels.astype(np.int32)
        prompt = prompt.astype(np.float32)

        x = np.swapaxes(x, 0, 1)
        prompt = np.swapaxes(prompt, 0, 1)
        # Select only sleep periods
        w_edge_mins = 30
        nw_idx = np.where(y != stage_dict["W"])[0]
        start_idx = nw_idx[0] - (w_edge_mins * 2)
        end_idx = nw_idx[-1] + (w_edge_mins * 2)
        if start_idx < 0: start_idx = 0
        if end_idx >= len(y): end_idx = len(y) - 1
        select_idx = np.arange(start_idx, end_idx + 1)
        logger.info("Data before selection: {}, {}".format(x.shape, y.shape))
        x = x[select_idx]
        y = y[select_idx]
        prompt = prompt[select_idx]
        logger.info("Data after selection: {}, {}".format(x.shape, y.shape))

        # Remove movement and unknown
        move_idx = np.where(y == stage_dict["MOVE"])[0]
        unk_idx = np.where(y == stage_dict["UNK"])[0]
        if len(move_idx) > 0 or len(unk_idx) > 0:
            remove_idx = np.union1d(move_idx, unk_idx)
            logger.info("Remove irrelavant stages")
            logger.info("  Movement: ({}) {}".format(len(move_idx), move_idx))
            logger.info("  Unknown: ({}) {}".format(len(unk_idx), unk_idx))
            logger.info("  Remove: ({}) {}".format(len(remove_idx), remove_idx))
            logger.info("  Data before removal: {}, {}".format(x.shape, y.shape))
            select_idx = np.setdiff1d(np.arange(x.shape[0]), remove_idx)
            x = x[select_idx]
            y = y[select_idx]
            prompt = prompt[select_idx]
            logger.info("  Data after removal: {}, {}".format(x.shape, y.shape))

        # Save
        filename = ntpath.basename(psg_fnames[i]).replace("-PSG.edf", ".npz")
        save_dict = {
            "x": x,
            "y": y,
            "x_prompt": prompt,
            "fs": sampling_rate,
            "ch_label": select_ch,
            "start_datetime": start_datetime,
            "file_duration": file_duration,
            "epoch_duration": epoch_duration,
            "n_all_epochs": n_epochs,
            "n_epochs": len(x),
        }
        np.savez(os.path.join(args.output_dir, filename), **save_dict)

        logger.info("\n=======================================\n")


if __name__ == "__main__":
    main()