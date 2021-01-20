# python3.6 tools/extract_features_hrnet.py --cfg experiments/cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml --testModel pretrained/hrnetv2_w18_imagenet_pretrained.pth
# python3.6 tools/extract_features_hrnet.py --cfg experiments/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml --testModel pretrained/hrnetv2_w64_imagenet_pretrained.pth

python3.6 tools/extract_features_hrnet_2.py --cfg experiments/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml --testModel pretrained/hrnetv2_w64_imagenet_pretrained.pth
# python3.6 tools/extract_features_hrnet.py --cfg experiments/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100_2.yaml --testModel pretrained/hrnetv2_w64_imagenet_pretrained.pth