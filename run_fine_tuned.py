import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-pre_trained_slimsam", type=str)
parser.add_argument("-fine_tuned_slimsam", type=str)
args = parser.parse_args()

def main():
  SlimSAM_model = torch.load(args.fine_tuned_slimsam)
  SlimSAM_model2 = torch.load(args.pre_trained_slimsam)
  original_state_dict = SlimSAM_model['model'] 
  modified_state_dict = {}
  for key, value in original_state_dict.items():
      if key.startswith('image_encoder.'):
          modified_key = 'image_encoder.module.' + key[len('image_encoder.'):]
      else:
          modified_key = key
      modified_state_dict[modified_key] = value
  
  SlimSAM_model2.load_state_dict(modified_state_dict)  
  print(SlimSAM_model2)

if __name__ == '__main__':
  main()

