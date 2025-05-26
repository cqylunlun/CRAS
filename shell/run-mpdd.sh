datapath=/root/cqy/dataset/MPDD
classes=('bracket_black' 'bracket_brown' 'bracket_white' 'connector' 'metal_plate' 'tubes')
flags=($(for class in "${classes[@]}"; do echo '-d '"${class}"; done))

clear
cd ..
python main.py \
    --gpu 0 \
    --seed 0 \
    --test ckpt \
  net \
    -b wideresnet50 \
    -le layer2 \
    -le layer3 \
    --pretrain_embed_dimension 1536 \
    --target_embed_dimension 1536 \
    --patchsize 3 \
    --meta_epochs 100 \
    --eval_epochs 1 \
    --dsc_layers 3 \
    --pre_proj 1 \
    --noise 0.015 \
    --k 0.3 \
    --limit -1 \
  dataset \
    --setting multi \
    --batch_size 32 \
    --resize 329 \
    --imagesize 288 "${flags[@]}" mpdd $datapath