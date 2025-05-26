datapath=/root/cqy/dataset/VisA
classes=('candle' 'capsules' 'cashew' 'chewinggum' 'fryum' 'macaroni1' 'macaroni2' 'pcb1' 'pcb2' 'pcb3' 'pcb4' 'pipe_fryum')
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
    --imagesize 288 "${flags[@]}" visa $datapath