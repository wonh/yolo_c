
#python train_multi.py --model_def config/spp.cfg --data_config config/3w8r+cds+pinp123+sr.data --batch_size 40 -mn spp_3w8r+cds+pinp123+sr
horovodrun -np 4 -H localhost:4 python train_hvd.py --model_def config/spp.cfg --data_config config/3w8r+cds+pinp123+sr_2.data --batch_size 10 -mn spp_3w8r+cds+pinp123+sr_2
