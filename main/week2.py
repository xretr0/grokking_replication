import torch
import transformer_lens
import pickle
# import time
from typing import Any
import os


P = 113


def load_model_to_pickle() -> None:
    """Loads the model into a pickle file. To save on time presumably"""
    # Load a model (eg GPT-2 Small)
    model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")

    with open('data/model_backup.pickle', 'wb') as out_f:
        pickle.dump(model, out_f)


def pickle_the_variable(**kwargs):
    """
    >>> x = 'blah'
    >>> y = 'stuff'
    >>> pickle_the_variable(x=x, y=y)
    >>> x1, y1 = unpickle_variables(['x', 'y'])
    >>> x == x1
    True
    >>> y == y1
    True

    :param kwargs:
    :return:
    """
    # timestamp = time.time_ns()  # int
    global_path = os.path.dirname(__file__)
    for key, value in kwargs.items():
        # path_to_file = os.path.join(global_path, f'{key}_{timestamp}.pickle')
        path_to_file = os.path.join(global_path, "data", f'{key}.pickle')
        with open(path_to_file, 'bw') as out_f:
            pickle.dump(value, out_f)


def unpickle_variables(variable_names_to_load: list[str]) -> list[Any]:
    list_to_return = []
    global_path = os.path.dirname(__file__)

    for variable_name in variable_names_to_load:
        file_name = f'{variable_name}.pickle'
        path_to_file = os.path.join(global_path, "data", file_name)
        with open(path_to_file, 'rb') as in_f:
            var = pickle.load(in_f)
            list_to_return.append(var)
    return list_to_return


def get_all_data() -> torch.Tensor:
    """
    # >>> P = 113
    # >>> get_all_data()
    """
    global P

    list_tmp = []
    for a in range(P):
        for b in range(P):
            sublist = [a, b, P, (a + b) % P]  # P is the encoding of the '='
            list_tmp.append(sublist)
    tensor_to_return = torch.tensor(list_tmp)
    return tensor_to_return


# def get_random_data(num=4000):
#     all_data =


def main() -> None:
    global P

    config_dict = {
        'd_model': 128,
        'd_head': 32,
        'n_layers': 1,
        'n_ctx': 4,  # for 'ab={thing_we_care_about}'
        'n_heads': 4,
        'd_mlp': 512,
        'd_vocab': P + 1,
        'act_fn': 'relu',
        'use_attn_result': True,
        'use_split_qkv_input': True,
        'use_hook_mlp_in': True,
        'use_attn_in': True,
        'normalization_type': None,
        'seed': 192,  # Set by us
        'positional_embedding_type': 'standard',
    }
    model = transformer_lens.HookedTransformer(config_dict)
    input_data = torch.tensor([[1, 2, 3]])
    predictions = model(input_data)[0, -1, :]
    print(1, predictions)


if __name__ == "__main__":
    main()


