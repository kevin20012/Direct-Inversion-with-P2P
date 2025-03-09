import os 
import numpy as np
import argparse
import json
from PIL import Image
import torch
import random
from models.p2p.p2p_editor import P2PEditor

def mask_decode(encoded_mask,image_shape=[512,512]):
    length=image_shape[0]*image_shape[1]
    mask_array=np.zeros((length,))
    
    for i in range(0,len(encoded_mask),2):
        splice_len=min(encoded_mask[i+1],length-encoded_mask[i])
        for j in range(splice_len):
            mask_array[encoded_mask[i]+j]=1
            
    mask_array=mask_array.reshape(image_shape[0], image_shape[1])
    # to avoid annotation errors in boundary
    mask_array[0,:]=1
    mask_array[-1,:]=1
    mask_array[:,0]=1
    mask_array[:,-1]=1
            
    return mask_array


def setup_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun_exist_images', action= "store_true") # rerun existing images
    parser.add_argument('--data_path', type=str, default="data") # the editing category that needed to run
    parser.add_argument('--output_path', type=str, default="output") # the editing category that needed to run

    # available editing methods combination: 
    # [ddim, null-text-inversion, negative-prompt-inversion, directinversion, inversion-free-editing] + 
    # [p2p, masactrl, pix2pix_zero, pnp]
    args = parser.parse_args()
    
    rerun_exist_images=args.rerun_exist_images
    data_path=args.data_path
    output_path=args.output_path
    
    
    
    # with open(f"{data_path}/mapping_file.json", "r") as f:
    #     editing_instruction = json.load(f)
    
    # for key, item in editing_instruction.items():
        
    #     if item["editing_type_id"] not in edit_category_list:
    #         continue
        
    original_prompt = "photo of bird"#item["original_prompt"].replace("[", "").replace("]", "")
    editing_prompt = "photo of frog"#item["editing_prompt"].replace("[", "").replace("]", "")
    image_path = "./img/bird.jpg"#os.path.join(f"{data_path}/annotation_images", item["image_path"])
    editing_instruction = "Make the bird to frog"#item["editing_instruction"]
    blended_word = ["bird", "frog"]#item["blended_word"].split(" ") if item["blended_word"] != "" else []
    # mask = Image.fromarray(np.uint8(mask_decode(item["mask"])[:,:,np.newaxis].repeat(3,2))).convert("L")
    mask = Image.fromarray(np.uint8(mask_decode([1 for i in range(512*512)])[:,:,np.newaxis].repeat(3,2))).convert("L")

    present_image_save_path=image_path.replace(data_path, os.path.join(output_path,"directinversion+p2p"))
    print(present_image_save_path)
    print(os.path.exists(present_image_save_path), rerun_exist_images)
    if ((not os.path.exists(present_image_save_path)) or rerun_exist_images):
        print(f"editing image [{image_path}] with [direct+p2p]")
        setup_seed()
        torch.cuda.empty_cache()
            
        p2p_editor=P2PEditor(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),num_ddim_steps=50)
        edited_image = p2p_editor("directinversion+p2p",
                                image_path=image_path,
                            prompt_src=original_prompt,
                            prompt_tar=editing_prompt,
                            guidance_scale=7.5,
                            cross_replace_steps=0.4,
                            self_replace_steps=0.6,
                            blend_word=(((blended_word[0], ),
                                        (blended_word[1], ))) if len(blended_word) else None,
                            eq_params={
                                "words": (blended_word[1], ),
                                "values": (2, )
                            } if len(blended_word) else None,
                            proximal="l0",
                            quantile=0.75,
                            use_inversion_guidance=True,
                            recon_lr=1,
                            recon_t=400,
                            )
            
        if not os.path.exists(os.path.dirname(present_image_save_path)):
            os.makedirs(os.path.dirname(present_image_save_path))
        edited_image.save(present_image_save_path)
        
        print(f"finish")
            
    else:
        print(f"skip image [{image_path}] with [direct+p2p]")