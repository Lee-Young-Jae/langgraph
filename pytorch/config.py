class Config:
    def __init__(self):
        self.bb = 'swin_v1_base'  # backbone
        self.lateral_channels_in_collection = [128, 256, 512, 1024]  # 채널 크기
        self.auxiliary_classification = False
        self.squeeze_block = None
        self.ender = False
        self.refine = None
        self.freeze_bb = False
        self.mul_scl_ipt = None
        self.cxt = []
        self.ms_supervision = False
        self.out_ref = False
        self.dec_ipt = False
        self.dec_ipt_split = False
        self.dec_blk = 'BasicDecBlk'
        self.lat_blk = 'BasicLatBlk'
        self.batch_size = 1
        self.size = [1024, 1024]  # 입력 이미지 크기 