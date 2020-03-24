import torch
import pickle

# checkpoint = torch.load('/root/pytorch_mobilenetV2/experiments/FPN_2/checkpoints/model_test_best.pth.tar')
# checkpoint = torch.load('/root/mobilenet_maskrcnn/output_1/model_final.pth')
checkpoint=torch.load('/root/pytorch_mobilenetV2/experiments/FPN_2/checkpoints/model_test_best_clear.pth.tar')
# checkpoint = torch.load('/root/maskrcnn-benchmark/demo/R-101.pkl')，原始的pkl文件只有model,详见load_resnet_c2_format
# checkpoint.pop('optimizer')
# checkpoint.pop('epoch')
# checkpoint.pop('best_top1')
# checkpoint['model'] = checkpoint.pop('state_dict')
# checkpoint.pop('iteration')
#torch.save(checkpoint, '/root/pytorch_mobilenetV2/experiments/FPN_2/checkpoints/model_test_best_clear.pth.tar')
pass



# with open('/root/maskrcnn-benchmark/demo/R-101.pkl', 'rb') as f:
#     info = pickle.load(f, encoding="latin1")
#     # a = f.read()
#     # print(a)
#     print(info)
#     pass
