import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel

#from sentence_transformers import SentenceTransformer, util

from src.table_encoder.gat import GATEncoder


class MatchingModel(nn.Module):
    def __init__(self, bert_dir='bert-base-uncased', do_lower_case=True, bert_size=768, gnn_output_size=300):
        super().__init__()
        self.res_size = 512
        self.tokenizer = BertTokenizer.from_pretrained(bert_dir, do_lower_case=True)
        self.bert = BertModel.from_pretrained(bert_dir)


        self.res_bert = nn.Sequential(
            
            nn.Linear(bert_size, self.res_size),
            nn.LayerNorm(self.res_size),
            nn.ReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(self.res_size, bert_size),
        )
        """
        self.projection = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(bert_size * 2, bert_size),
        )
        """

        self.gnn = GATEncoder(input_dim=300, output_dim=gnn_output_size, hidden_dim=300, layer_num=4,
                              activation=nn.LeakyReLU(0.2))

        self.project_table = nn.Sequential(
            nn.Linear(gnn_output_size, 300),
            nn.LayerNorm(300)
        )

        self.dim_reduction = nn.Sequential(
            nn.Linear(1200, 1200),
            nn.Tanh(),
        )

        self.regression = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(bert_size + 1200, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
        )

    def forward(self, table, query, dgl_g, t_feat, q_feat):
        """table retrieval"""
        bert_rep = self.text_matching(table, query)
        bert_rep_max_res = self.res_bert(bert_rep) + bert_rep

        gnn_rep = self.text_table_matching(dgl_g, t_feat, q_feat)

        rep = torch.cat((bert_rep_max_res, gnn_rep), -1)

        score = self.regression(rep)

        return score

    def text_table_matching(self, dgl_graph, table_embs, query_emb):
        """text-table matching module"""
        creps = self.gnn(dgl_graph, table_embs)

        tmapping = self.project_table(creps)
        qmapping = query_emb.repeat(creps.shape[0], 1)

        hidden = torch.cat((tmapping, qmapping, tmapping - qmapping, tmapping * qmapping), 1)

        hidden = self.dim_reduction(hidden)

        hidden = torch.max(hidden, 0)[0]

        return hidden

    def tokenize(self, input_text):
        inputs = self.tokenizer.encode_plus(
            input_text,
            None,
            add_special_tokens=True,
            max_length = 128,
            padding='max_length',
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        return ids,mask,token_type_ids

    def text_matching(self, table, query):
        """text-text matching module"""

        ids1,mask1,token_type_ids1 = self.tokenize(" ".join(query))

        #print(table["caption"])
        #print(table["pgTitle"])
        #print(table["secondTitle"])
        table_str = table["caption"]
        if 'subcaption' in table:
            table_str += table["subcaption"]
            # tokens += ["[SEP]"]

        if 'pgTitle' in table:
            table_str += table["pgTitle"]
            # tokens += ["[SEP]"]

        if 'secondTitle' in table:
            table_str += table["secondTitle"]
            # tokens += ["[SEP]"]

        ids2,mask2,token_type_ids2 = self.tokenize(table_str)
        tokens_tensor = torch.tensor([ids1]).to("cuda")
        mask_tensor = torch.tensor([mask1]).to("cuda")
        token_type_tensor = torch.tensor([token_type_ids1]).to("cuda")

        tokens_table_tensor = torch.tensor([ids2]).to("cuda")
        mask_table_tensor = torch.tensor([mask2]).to("cuda")
        token_table_type_tensor = torch.tensor([token_type_ids2]).to("cuda")

        outputs1 = self.bert(tokens_tensor, mask_tensor, token_type_ids=token_type_tensor)
        outputs2 = self.bert(tokens_table_tensor, mask_table_tensor, token_type_ids=token_table_type_tensor)
        pooled_embedding_1 = outputs1[1][0]  # pooled output of the [CLS] token
        pooled_embedding_2 = outputs2[1][0]
        #print(outputs1[0].shape)
        #e_sim = torch.dot(pooled_embedding_1, pooled_embedding_2)
        #print("Current e_sim :", e_sim)
        return torch.maximum(pooled_embedding_1, pooled_embedding_2)
