
class Params:

    def __init__(self, 
                 ne=500, 
                 bsize=512, 
                 lr=0.001, 
                 reg_lambda=0.0, 
                 emb_dim=100, 
                 neg_ratio=20, 
                 dropout=0.4,  
                 save_each=50,  
                 dataname='wikidata12k',
                 granularity_dim=3,
                 valid_ratio=0.1,
                 test_ratio=0.1,
                 time_vec_dim=10,
                 time_negative_sampling=True,
                 hidden_dim=10,
                 pretrained_embedding_path='/data/tanhexiang/temporal_link_prediction/models/TransE/wikidata12k/500_512_0.001_0.0_90_20_0.4_10_50_0.9_500.chkpnt'):

        self.ne = ne
        self.bsize = bsize
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.emb_dim=emb_dim
        self.save_each = save_each
        self.neg_ratio = neg_ratio
        self.dropout = dropout
        self.dataname = dataname
        self.valid_ratio=valid_ratio
        self.test_ratio=test_ratio
        self.granularity_dim=granularity_dim
        self.pretrained_embedding_path=pretrained_embedding_path
        self.time_vec_dim=time_vec_dim
        self.hidden_dim=hidden_dim
        
    def str_(self):
        return str(self.ne) + "_" + str(self.bsize) + "_" + str(self.lr) + "_" + str(self.reg_lambda) + "_" + str(self.emb_dim) + "_" + str(self.neg_ratio) + "_" + str(self.dropout)  + "_" + str(self.save_each) + "_" 