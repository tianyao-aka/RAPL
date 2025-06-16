import torch
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer

class GTELargeEN:
    def __init__(self,
                 device,
                 normalize=True):
        self.device = device
        #! load from huggingface
        model_path = 'Alibaba-NLP/gte-large-en-v1.5'
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            unpad_inputs=True,
            torch_dtype=torch.bfloat16,  # Mixed precision
            attn_implementation='sdpa').to(device)
        
        #! load from disk
        # local_model_path = 'hf_models/gte-large-en-v1.5/gte-large-en-v1.5/'  # replace with actual path
        # print (f"Loading model from {local_model_path}")
        # self.tokenizer = AutoTokenizer.from_pretrained(local_model_path,local_files_only=True,trust_remote_code=True)
        # self.model = AutoModel.from_pretrained(
        #     local_model_path,
        #     trust_remote_code=True,
        #     local_files_only=True,
        #     unpad_inputs=True,
        #     torch_dtype=torch.bfloat16,  # Mixed precision
        #     attn_implementation='sdpa'
        # ).to(device)
        
        self.normalize = normalize

    @torch.no_grad()
    def embed(self, text_list):
        if len(text_list) == 0:
            return torch.zeros(0, 1024)
        
        batch_dict = self.tokenizer(
            text_list, max_length=8192, padding=True,
            truncation=True, return_tensors='pt').to(self.device)
        
        outputs = self.model(**batch_dict).last_hidden_state
        emb = outputs[:, 0]
        
        if self.normalize:
            emb = F.normalize(emb, p=2, dim=1)
        
        return emb.cpu()

    def __call__(self, q_text, text_entity_list, relation_list):
        q_emb = self.embed([q_text])
        entity_embs = self.embed(text_entity_list)
        relation_embs = self.embed(relation_list)
        
        return q_emb, entity_embs, relation_embs
    
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    text_encoder = GTELargeEN(device)
    print (text_encoder)
