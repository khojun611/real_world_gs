# python train.py -s ../gaussian-splatting/artifacts/02 --eval -r 2 --white_background 
# python train.py -s ../gaussian-splatting/artifacts/ceramic --eval -r 2
# python train.py -s ../mip-splatting/artifacts/yeonjeok --eval
# python train.py -s ../mip-splatting/artifacts/pouch --eval 
# python train.py -s ../mip-splatting/artifacts/mirror --eval 
# python train.py -s ../mip-splatting/artifacts/ceramic --eval 
# python train.py -s ../mip-splatting/artifacts/cup --eval 
# python train.py -s ../mip-splatting/artifacts/hinge --eval -r 8
# python train.py -s ../3DGS-DR/data/chrome_back --eval -r 4
# python train.py -s ./data/chrome_table --eval -r 2
# python train.py -s ./data/custom/ball_back --eval 
python train.py -s ./data/chrome_book -r 2 --eval --iterations 20000 --indirect_from_iter 10000 --volume_render_until_iter 0  --initial 1 --init_until_iter 3000