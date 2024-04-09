def test():
  MedSAM_CKPT_PATH = "/content/SlimSAM-50.pth"
  device = "cuda:0"

  import torch
  SlimSAM_model = torch.load(MedSAM_CKPT_PATH)
  SlimSAM_model.image_encoder = SlimSAM_model.image_encoder.module

  def forward(self, x):
      print(x)
      print("here")
      x = self.patch_embed(x)
      if self.pos_embed is not None:
          x = x + self.pos_embed

      for blk in self.blocks:
          x,qkv_emb,mid_emb,x_emb = blk(x)

      x = self.neck(x.permute(0, 3, 1, 2))

      return x

  import types
  funcType = types.MethodType
  SlimSAM_model.image_encoder.forward = funcType(forward, SlimSAM_model.image_encoder)
  print(SlimSAM_model.image_encoder)

if __name__ == '__main__':
    test()
