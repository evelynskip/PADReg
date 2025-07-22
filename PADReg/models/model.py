'''
Swin-Transformer code retrieved from:
https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation

TransMorph code retrieved from:
https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration

Original paper:
Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., ... & Guo, B. (2021).
Swin transformer: Hierarchical vision transformer using shifted windows.
arXiv preprint arXiv:2103.14030.

Chen, J., Frey, E. C., He, Y., Segars, W. P., Li, Y., & Du, Y. (2022). 
Transmorph: Transformer for unsupervised medical image registration. 
Medical image analysis, 82, 102615.
'''

import torch.nn as nn
import models.configs_TransMorph as configs
import models.TransMorph as TM
from models.Force_head import *
from models.force_emb import ForceEmbedding, CrossAttention

class PADReg(nn.Module):
    def __init__(self, config):
        '''
        TransMorph Model
        '''
        super(PADReg, self).__init__()
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = TM.SwinTransformer(patch_size=config.patch_size,
                                           in_chans=config.in_chans,
                                           embed_dim=config.embed_dim,
                                           depths=config.depths,
                                           num_heads=config.num_heads,
                                           window_size=config.window_size,
                                           mlp_ratio=config.mlp_ratio,
                                           qkv_bias=config.qkv_bias,
                                           drop_rate=config.drop_rate,
                                           drop_path_rate=config.drop_path_rate,
                                           ape=config.ape,
                                           spe=config.spe,
                                           rpe=config.rpe,
                                           patch_norm=config.patch_norm,
                                           use_checkpoint=config.use_checkpoint,
                                           out_indices=config.out_indices,
                                           pat_merg_rf=config.pat_merg_rf,
                                           )
        self.up0 = TM.DecoderBlock(embed_dim*8, embed_dim*4, skip_channels=embed_dim*4 if if_transskip else 0, use_batchnorm=False)
        self.up1 = TM.DecoderBlock(embed_dim*4, embed_dim*2, skip_channels=embed_dim*2 if if_transskip else 0, use_batchnorm=False)  # 384, 20, 20, 64
        self.up2 = TM.DecoderBlock(embed_dim*2, embed_dim, skip_channels=embed_dim if if_transskip else 0, use_batchnorm=False)  # 384, 40, 40, 64
        self.up3 = TM.DecoderBlock(embed_dim, embed_dim//2, skip_channels=embed_dim//2 if if_convskip else 0, use_batchnorm=False)  # 384, 80, 80, 128
        self.up4 = TM.DecoderBlock(embed_dim//2, config.reg_head_chan, skip_channels=config.reg_head_chan if if_convskip else 0, use_batchnorm=False)  # 384, 160, 160, 256
        self.c1 = TM.Conv2dReLU(2, embed_dim//2, 3, 1, use_batchnorm=False)
        self.c2 = TM.Conv2dReLU(2, config.reg_head_chan, 3, 1, use_batchnorm=False)
        self.reg_head = TM.RegistrationHead(
            in_channels=config.reg_head_chan,
            out_channels=2,
            kernel_size=3,
        )
        self.spatial_trans = TM.SpatialTransformer(config.img_size)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)

        self.stiff2flow = Stiff2Flow()
        self.frc_emb = ForceEmbedding(n_channels=embed_dim*8)

    def forward(self, x,  frc, m_ann=None):
        """
        Return:
            output_list [out, flow, pre_frc(optional)[B,C], o_ann(optional)[B,1,H,W,float32]]
        """
        source = x[:, 0:1, :, :]
        if self.if_convskip:
            x_s0 = x.clone()
            x_s1 = self.avg_pool(x)
            f4 = self.c1(x_s1)
            f5 = self.c2(x_s0)
        else:
            f4 = None
            f5 = None
        out_feats = self.transformer(x)

        if self.if_transskip:
            f1 = out_feats[-2]
            f2 = out_feats[-3]
            f3 = out_feats[-4]
        else:
            f1 = None
            f2 = None
            f3 = None
        """Force Fusion"""
        f_emb = self.frc_emb(frc)
        x1 = self.up0(out_feats[-1]*f_emb.unsqueeze(2).unsqueeze(3), f1)

        """Decoder"""
        x2 = self.up1(x1, f2)
        x3 = self.up2(x2, f3)
        x4 = self.up3(x3, f4)
        x5 = self.up4(x4, f5)

        stiff = self.reg_head(x5)
        flow = self.stiff2flow(stiff,frc)

        out = self.spatial_trans(source, flow)

        output_list = [out, flow]

        output_list.append(stiff)

        """Get deformed mask"""
        if m_ann is not None:
            m_ann_ = m_ann.to(torch.float32)
            o_ann = self.spatial_trans(m_ann_, flow, mode="nearest")
            output_list.append(o_ann)
        
        return output_list



CONFIGS = {
    'TransMorph': configs.get_2DTransMorph_config(),
    'TransMorph-No-Conv-Skip': configs.get_2DTransMorphNoConvSkip_config(),
    'TransMorph-No-Trans-Skip': configs.get_2DTransMorphNoTransSkip_config(),
    'TransMorph-No-Skip': configs.get_2DTransMorphNoSkip_config(),
    'TransMorph-Lrn': configs.get_2DTransMorphLrn_config(),
    'TransMorph-Sin': configs.get_2DTransMorphSin_config(),
    'TransMorph-No-RelPosEmbed': configs.get_2DTransMorphNoRelativePosEmbd_config(),
    'TransMorph-Large': configs.get_2DTransMorphLarge_config(),
    'TransMorph-Small': configs.get_2DTransMorphSmall_config(),
    'TransMorph-Tiny': configs.get_2DTransMorphTiny_config(),
}
