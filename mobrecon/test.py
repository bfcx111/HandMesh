import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mobrecon.build import build_model, build_dataset
from mobrecon.configs.config import get_cfg
from options.cfg_options import CFGOptions
from mobrecon.runner import Runner
import os.path as osp
from utils import utils
from utils.writer import Writer
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    # default_setup(cfg, args)
    return cfg


def main(args):
    # get config
    cfg = setup(args)

    # device

    if -1 in cfg.TRAIN.GPU_ID or not torch.cuda.is_available():
        device = torch.device('cpu')
        print('CPU mode')
    elif len(cfg.TRAIN.GPU_ID) == 1:
        device = torch.device('cuda', cfg.TRAIN.GPU_ID[0])
        print('CUDA ' + str(cfg.TRAIN.GPU_ID) + ' Used')
    else:
        raise Exception('Do not support multi-GPU training')


    exec('from mobrecon.models.{} import {}'.format(cfg.MODEL.NAME.lower(), cfg.MODEL.NAME))
    exec('from mobrecon.datasets.{} import {}'.format(cfg.TRAIN.DATASET.lower(), cfg.TRAIN.DATASET))
    exec('from mobrecon.datasets.{} import {}'.format(cfg.VAL.DATASET.lower(), cfg.VAL.DATASET))

    breakpoint()
    # model
    model = build_model(cfg).to(device)

    # resume
    model_path = '/home/xxx/work/otherwork/HandMesh/mobrecon/out/checkpoint_best.pt'
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    image_fp = '/home/xxx/work/otherwork/Hand_detect/MeshTransformer/samples/data'
    image_files = [os.path.join(image_fp, i) for i in os.listdir(image_fp)]
    # bar = Bar(colored("DEMO", color='blue'), max=len(image_files))
    import cv2
    import time
    from utils.vis import base_transform
    from utils.read import save_mesh
    from manopth.manolayer import ManoLayer

    def save_obj(v, f, file_name='output.obj'):
        obj_file = open(file_name, 'w')
        for i in range(len(v)):
            obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
        for i in range(len(f)):
            obj_file.write('f ' + str(f[i][0]+1) + '/' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + '/' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '/' + str(f[i][2]+1) + '\n')
        obj_file.close()
    mano_path = '/home/xxx/work/EasyMocapPublic/data/bodymodels/manov1.2/'
    mano_layer = ManoLayer(mano_root=mano_path, flat_hand_mean=False, use_pca=False) # load right hand MANO model
    faces = mano_layer.th_faces.cpu().numpy()
    with torch.no_grad():
        for step, image_path in enumerate(image_files):
            image_name = image_path.split('/')[-1].split('_')[0]
            image = cv2.imread(image_path)[..., ::-1]
            image = cv2.resize(image, (128, 128))
            input = torch.from_numpy(base_transform(image, size=128)).unsqueeze(0).to(device)
            
            torch.cuda.synchronize()
            start = time.time()
            out = model(input)
            torch.cuda.synchronize()
            end = time.time()
            print('time:' ,end - start)

            pred = out['verts']#[0]
            vertex = (pred[0].cpu() * 0.2).numpy()
            #print('std',self.std)std tensor(0.2000)
            save_obj(vertex, faces, os.path.join('/home/xxx/work/otherwork/HandMesh/mobrecon/output', image_name + '_mesh.obj'))






if __name__ == "__main__":

    args = CFGOptions().parse()
    # args.exp_name = 'test'
    # args.config_file = 'mobrecon/configs/mobrecon_ds.yml'
    main(args)



# CUDA_VISIBLE_DEVICES=0 python -m mobrecon.test --exp_name mrc_ds --config_file mobrecon/configs/mobrecon_ds_test.yml
