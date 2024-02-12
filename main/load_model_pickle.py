import transformer_lens
import pickle

# Load a model (eg GPT-2 Small)
model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")

with open('data/model_backup.pickle', 'wb') as out_f:
    pickle.dump(model, out_f)
