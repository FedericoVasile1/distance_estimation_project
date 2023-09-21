import os

if __name__ == "__main__":
    FREEZE_AT = [3, 4]
    RNN = [True, False]
    IMG_FEATMAP = [True, False]
    for freeze_at in FREEZE_AT:
        for rnn in RNN:
            for img_featmap in IMG_FEATMAP:
                cmd = "python main.py --model zhu --backbone resnet34 " \
                      "--regressor simple_roi --batch_size 4 --input_h_w 720 1280 " \
                      "--lr 5e-05 --loss l1 " \
                      "--ds_path /home/federico/temp/ellis/MOTSynth " \
                      "--annotations_path /home/federico/temp/ellis/annotations_clean " \
                      f"--freeze_at {freeze_at} " 
                cmd += "--rnn " if rnn else ""
                cmd += "--img_featmap " if img_featmap else ""
                
                os.system(cmd)