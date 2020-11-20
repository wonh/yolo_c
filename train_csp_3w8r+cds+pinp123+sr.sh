#python train_multi.py --model_def config/csp.cfg --data_config config/3w8r+cds+pinp123+sr.data --batch_size 32 -mn csp_3w8r+cds+pinp123+sr
horovodrun -np 3 -H localhost:3 python train_hvd.py --model_def config/csp.cfg --data_config config/3w8r+cds+pinp123+sr_2.data --batch_size 8 -mn csp_3w8r+cds+pinp123+sr_2
