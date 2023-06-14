import os
import torch
from collections import OrderedDict
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms as transforms
import torchvision.models as models
from timm.models import create_model
import vision_transformer as vits


has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark = True



class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


def get_model(args, nclasses, resume_pth, device='cuda:0'):
    if args.network == 'resnet50':
        model = models.resnet50(pretrained=False)
        in_features = 2048
        model.fc = torch.nn.Linear(in_features=in_features, out_features=nclasses, bias=True)
    elif args.network == 'resnet18':
        model = models.resnet18(pretrained=False)
        in_features = 512
        model.fc = torch.nn.Linear(in_features=in_features, out_features=nclasses, bias=True)
    elif args.network == "convnext_tiny":
        model = models.convnext_tiny(pretrained=False )
        in_features = 768
        model.classifier = torch.nn.Sequential(torch.nn.LayerNorm([in_features, 1, 1], elementwise_affine=True),
                                         torch.nn.Flatten(),
                                         torch.nn.Linear(in_features=in_features, out_features=nclasses, bias=True)
                                        )
    _ = resume_checkpoint(model, resume_pth, log_info=True, location=device)
    model.to(device)
    return model


def get_timm_model(model_name, nclasses, resume_pth, device='cuda:0'):
    pretrained = True if not resume_pth else False
    model = create_model(
        model_name,  # e.g efficientnet_b2
        num_classes=nclasses,
        in_chans=3,
        pretrained=pretrained,
        checkpoint_path=resume_pth,
    )
    model = model.to(device)
    return model


def resume_checkpoint(model, checkpoint_path, optimizer=None, loss_scaler=None, log_info=True, location='cpu'):
    resume_epoch = None
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=location)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            if log_info:
                print('Restoring model state from checkpoint...')
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)

            if optimizer is not None and 'optimizer' in checkpoint:
                if log_info:
                    print('Restoring optimizer state from checkpoint...')
                optimizer.load_state_dict(checkpoint['optimizer'])

            if loss_scaler is not None and loss_scaler.state_dict_key in checkpoint:
                if log_info:
                    print('Restoring AMP loss scaler state from checkpoint...')
                loss_scaler.load_state_dict(checkpoint[loss_scaler.state_dict_key])

            if 'epoch' in checkpoint:
                resume_epoch = checkpoint['epoch']
                if 'version' in checkpoint and checkpoint['version'] > 1:
                    resume_epoch += 1  # start at the next epoch, old checkpoints incremented before save

            if log_info:
                print("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
        else:
            model.load_state_dict(checkpoint)
            if log_info:
                print("Loaded checkpoint '{}'".format(checkpoint_path))
        return resume_epoch
    else:
        print("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def load_pretrained_weights(model, pretrained_weights, checkpoint_key, model_name, patch_size):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        url = None
        if model_name == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif model_name == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        elif model_name == "xcit_small_12_p16":
            url = "dino_xcit_small_12_p16_pretrain/dino_xcit_small_12_p16_pretrain.pth"
        elif model_name == "xcit_small_12_p8":
            url = "dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_pretrain.pth"
        elif model_name == "xcit_medium_24_p16":
            url = "dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_pretrain.pth"
        elif model_name == "xcit_medium_24_p8":
            url = "dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_pretrain.pth"
        elif model_name == "resnet50":
            url = "dino_resnet50_pretrain/dino_resnet50_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
        else:
            print("There is no reference weights available for this model => We use random weights.")


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, img_list, transform, df):
        super(MyDataset, self).__init__()
        self.img_list = img_list
        self.transform = transform
        self.df = df

    def __len__(self):
        return len(self.img_list)

    def del_row(self, idx):
        if not self.df.empty:
            self.df.drop([idx], axis=0, inplace=True)

    def get_df(self):
        if not self.df.empty:
            self.df.reset_index(drop=True)
        return self.df

    def __getitem__(self, idx):
        try:
            img = Image.open(self.img_list[idx])
            if img.mode == 'L':
                img = img.convert('RGB')
            return self.transform(img)
        except Exception as e:
            print(f"ERROR: {e} on {self.img_list[idx]}")
            if not self.df.empty:
                self.del_row(idx)


def single_clf_inference(model, image, input_size=224, crop_size=128, device='cuda:0'):
    tfms = transforms.Compose([transforms.ToTensor(),
                               transforms.Resize(size=(input_size, input_size)),
                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                               ])

    model.eval()
    batch = tfms(image).unsqueeze(0)

    if crop_size < input_size:
        batch = crop_center(batch, crop_size)
    batch = batch.to(device=device)
    with torch.no_grad():
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                output = model(batch)
        else:
            output = model(batch)
        pred = torch.argmax(output, 1).view(-1).data.cpu().numpy().tolist()[0]
    return pred


def crop_center(images_tensor, crop_size):
    # cropping center and resizing
    height = images_tensor.shape[-1]
    start = height // 2 - crop_size // 2
    end = start + crop_size
    noraml_size = images_tensor.size()
    images_tensor = images_tensor[:, :, start:end, start:end]
    images_tensor = transforms.functional.resize(images_tensor, noraml_size[-2:])
    return images_tensor


def get_dino_model(pretrained_weights, device='cuda', model_key='teacher', lin_key='state_dict', arch='vit_small',
                   patch_size=16, n=4, avg_pool=False):
    

    model = vits.__dict__[arch](patch_size, num_classes=0)
    embed_dim = model.embed_dim * (n + int(avg_pool))
    model.to(device)

    # load weights to evaluate
    utils.load_pretrained_weights(model, '', model_key, arch, patch_size)

    linear_classifier = LinearClassifier(embed_dim, num_labels=2)
    linear_classifier = linear_classifier.to(device)

    state_dict = torch.load(pretrained_weights, map_location="cpu")
    if lin_key and lin_key in state_dict:
        print(f"Take key {'state_dict'} in provided checkpoint dict")
        state_dict = state_dict['state_dict']
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = linear_classifier.load_state_dict(state_dict, strict=False)
    print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))

    model.eval()
    linear_classifier.eval()
    return model, linear_classifier


def single_vit_inference(model, linear_classifier, image, input_size=384, crop_size=300, device='cuda:0',
                         avgpool=False):
    tfms = transforms.Compose([transforms.ToTensor(),
                               transforms.Resize(size=(input_size, input_size)),
                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                               ])

    model.eval()
    linear_classifier.eval()

    inp = tfms(image).unsqueeze(0)

    if crop_size < input_size:
        inp = crop_center(inp, crop_size)
    inp = inp.to(device=device)
    with torch.no_grad():
        intermediate_output = model.get_intermediate_layers(inp, 4)
        #print(intermediate_output[0].shape)
        output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
        if avgpool:
            output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)),
                               dim=-1)
            output = output.reshape(output.shape[0], -1)

        output = linear_classifier(output)
        pred = torch.argmax(output, 1).view(-1).data.cpu().numpy().tolist()[0]
    return pred



def get_model_ensemble(args, histo_models_names, device):
    histo_models_dict = {}
    for indx, histo_pth in enumerate(histo_models_names):
        histo_model = get_timm_model(args.network, args.histo_classes, histo_pth, device)
        histo_models_dict[f'conv_{indx}'] = histo_model

    model, linear_classifier = get_dino_model(f'{args.model_dir}/dino_checkpoint.pth.tar', device=device,
                                            model_key='teacher', lin_key='state_dict',
                                            arch='vit_small', patch_size=16, n=4, avg_pool=False)

    histo_models_dict['dino'] = (model, linear_classifier)
    return histo_models_dict

