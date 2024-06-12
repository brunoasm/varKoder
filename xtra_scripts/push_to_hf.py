from fastai.vision.all import load_learner
from huggingface_hub import push_to_hub_fastai
from timm.models.hub import push_to_hf_hub
from timm import create_model
import copy

#push to fastai
learn = load_learner("varkoder_trained_model_ML/trained_model.pkl")
push_to_hub_fastai(learner=learn, repo_id="vit_large_patch32_224.NCBI_SRA")

#start a timm model from scratch and update parameters based on fastai model
pretrained_cfg = {'hf_hub_id': 'brunoasm/vit_large_patch32_224.NCBI_SRA', 'source': 'hf-hub', 'architecture': 'hf-hub:timm/vit_large_patch32_224', 'tag': 'NCBI_SRA', 'custom_load': False, 'input_size': [3, 224, 224], 'fixed_input_size': True, 'interpolation': 'nearest', 'crop_pct': 1, 'crop_mode': 'center', 'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5], 'num_classes': 0, 'pool_size': None, 'first_conv': 'patch_embed.proj', 'classifier': 'head'}

mdl = create_model("timm/vit_large_patch32_224.orig_in21k",pretrained=True, num_classes=len(learn.dls.vocab))

state_dict = copy.deepcopy(mdl.state_dict())
for f_k in learn.state_dict().keys():
    for k in mdl.state_dict().keys():    
        if f_k.find(k) >= 0 and state_dict[k].shape == learn.state_dict()[f_k].shape:
            state_dict[k] = learn.state_dict()[f_k]
            break
mdl.load_state_dict(state_dict, strict=False)


model_cfg = dict(label_names=list(learn.dls.vocab))
push_to_hf_hub(model=mdl, repo_id="vit_large_patch32_224.NCBI_SRA", model_config=model_cfg)

