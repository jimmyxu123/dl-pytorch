import torch
import torch.nn as nn
import onnx
import os
from sppnet import *
from torchvision import transforms
from loguru import logger
from PIL import Image
from onnxsim import simplify

class SppInfer:
    def __init__(self,device='cpu') -> None:
        self.device = device
        # construct model and load pretrained model weights
        self.model = SppNet()
        # pre-process transform
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
        self.test_transform = transforms.Compose([transforms.ToTensor(),
                                            normalize]
                                            )
        
    def pre_process(self, input_data=None):
        self.input_tensor = self.test_transform(input_data).unsqueeze(0).to(device)
    
    def inference(self):
        #try:
        print(self.input_tensor.size())
        self.predictions = self.model(self.input_tensor).to(self.device)
        logger.debug('Finish Inference')
        #except:
        #    logger.error('Inference failed !')
        
    def write_result(self, output_path=None):
        np.save(output_path, self.predictions.squeeze().detach().cpu().numpy())

'''
    def post_process(self):
        # softmax probability
        prob = torch.softmax(self.predictions, 1)
        # results
        topk_prob, topk_id = torch.topk(prob, 5)
        logger.debug(f"TOP 5 pred prob Results: ")
        for i in range(5):
            logger.debug("%s: %.4f"%(self.categories[topk_id[0][i]].strip(), topk_prob[0][i]))
'''

if __name__ == '__main__':
    device = 'cpu'
    export_onnx = True
    onnx_sim = True
    count_ops_flops = True

    # prepare inference engine
    infer_engine = SppInfer()

    # prepare model input
    img_path = './cat.jpg'
    input_image = Image.open(img_path)
    infer_engine.pre_process(input_data=input_image)

    # model inference
    logger.debug('Start model inference ...... ')
    infer_engine.inference()
    logger.debug('End model inference ! ')

    # post-process
    #infer_engine.post_process()

    # write out result
    output_path = './predictions.npy'
    infer_engine.write_result(output_path)

    # export onnx
    # onnx_path = '../../transformer/vit.onnx'
    onnx_path = './spp.onnx'
    if export_onnx:
        logger.debug('Start export onnx ...... ')
        try:
            torch.onnx.export(infer_engine.model.to(device), 
                                infer_engine.input_tensor.to(device), 
                                onnx_path, 
                                export_params=True, 
                                opset_version=11, 
                                do_constant_folding=False,
                                input_names=['input'],
                                output_names=['output']
                            )
            #np.save('../output/original/resnet18_input.npy', infer_engine.input_tensor.cpu().numpy(), allow_pickle=True)
        except:
            logger.error('export onnx failed !')
            import traceback
            traceback.print_exc()
        logger.debug('End export onnx !')

    if onnx_sim and os.path.isfile(onnx_path):
        # load onnx model
        onnx_model = onnx.load(onnx_path)
        # convert model
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, onnx_path.replace('spp.onnx', 'spp_simplified.onnx'))

    # model statistics
    if count_ops_flops:
        from thop import profile
        logger.debug('Start count model flops and params ...... ')
        logger.debug('input_tensor.shape = ', infer_engine.input_tensor.shape)
        flops, params = profile(infer_engine.model.to(device), inputs=(infer_engine.input_tensor.to(device),))
        logger.debug('flops = %d, params = %d'%(flops, params))
        logger.debug('End count model flops and params !')
    
    ishape = infer_engine.input_tensor.shape
    save_json_str = """{"input_shape":"[%d,%d,%d,%d]", "flops":%d, "params":%d}"""%(ishape[0], ishape[1], ishape[2], ishape[3], flops, params)
    with open(onnx_path.replace('_original.onnx', '_info.txt'), 'w+', encoding='utf-8') as f:
        f.write(save_json_str)



