python preprocess_senorita.py --task "inpainting"
CUDA_VISIBLE_DEVICES=0,1,2,3 python inference_vie_score_filter.py --task "inpainting"

python preprocess_senorita.py --task "outpainting"
CUDA_VISIBLE_DEVICES=0,1,2,3 python inference_vie_score_filter.py --task "outpainting"

python preprocess_senorita.py --task "local_style_transfer"
CUDA_VISIBLE_DEVICES=0,1,2,3 python inference_vie_score_filter.py --task "local_style_transfer"