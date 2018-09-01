# SSD_tf_simple_implemented

This is a pure tf 1.4.0 implemention of SSD: Single Shot MultiBox Detector. Modified according to the Keras version https://github.com/rykov8/ssd_keras .

## Dependency：

    python3.5
    tensorflow 1.4
    opencv2
    h5py (optional if you delete one code block in ssd.py)
    pickle
    numpy
    
## How to predict:

    python3 ssd.py --mode=eval --images_dir=path/to/images --label_file=gt_pascal.pkl --checkpoint=path/to/checkpoint

*Parameters are all optional. The default values are in ssd_utils.py

## How to train:

    python3 ssd.py --mode=train --images_dir=path/to/images --label_file=gt_pascal.pkl --checkpoint=path/to/checkpoint

*Parameters are all optional. The default values are in ssd_utils.py

## How to prepare data:
Two ways:
1.Format your data as the Pascal VOC format and run:
python3 tools/get_data_from_XML.py
You may need to modify some path in tools/get_data_from_XML.py

2.Run tools/get_data_from_txt.py if your data is txt files. Path modification is also needed.

3.Remember to run tools/check_gt_pascal.py to see whether gt_pascal.pkl is correct.
