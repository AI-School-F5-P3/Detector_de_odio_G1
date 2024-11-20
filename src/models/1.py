import torch

# Guardar los modelos entrenados
torch.save(model_bert.state_dict(), 'bert_model.pth')
torch.save(model_roberta.state_dict(), 'roberta_model.pth')
