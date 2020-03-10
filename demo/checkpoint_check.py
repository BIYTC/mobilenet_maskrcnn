import torch
import pickle

#checkpoint = torch.load('/root/maskrcnn-benchmark-1/output_3/model_final.pth')
#checkpoint = torch.load('/root/maskrcnn-benchmark-1/output_3/model_final_clear.pth')
# checkpoint = torch.load('/root/maskrcnn-benchmark/demo/R-101.pkl')，原始的pkl文件只有model,详见load_resnet_c2_format
# checkpoint.pop('optimizer')
# checkpoint.pop('scheduler')
# checkpoint.pop('iteration')
# torch.save(checkpoint,'/root/maskrcnn-benchmark-1/output_3/model_final_clear.pth')
pass



with open('/root/maskrcnn-benchmark/demo/R-101.pkl', 'rb') as f:
    info = pickle.load(f, encoding="latin1")
    # a = f.read()
    # print(a)
    print(info)
    pass
