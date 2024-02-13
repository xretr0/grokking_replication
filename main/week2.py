import torch
import transformer_lens
import pickle
# import time
from typing import Any, Optional
import os


P = 113


def load_model_to_pickle() -> None:
    """Loads the model into a pickle file. To save on time presumably"""
    # Load a model (eg GPT-2 Small)
    model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")

    with open('data/model_backup.pickle', 'wb') as out_f:
        pickle.dump(model, out_f)


def pickle_the_variable(**kwargs) -> None:
    """
    Pickles the variable <value> into a file ./data/<key>.pickle.

    >>> x = 'blah'
    >>> y = 'stuff'
    >>> pickle_the_variable(x=x, y=y)
    >>> x1, y1 = unpickle_variables(['x', 'y'])
    >>> x == x1
    True
    >>> y == y1
    True

    :param kwargs: Pairs of <key> - filename and <value> - variable to pickle
    :return: None
    """
    # timestamp = time.time_ns()  # int
    global_path = os.path.dirname(__file__)
    for key, value in kwargs.items():
        # path_to_file = os.path.join(global_path, f'{key}_{timestamp}.pickle')
        path_to_file = os.path.join(global_path, "data", f'{key}.pickle')
        with open(path_to_file, 'bw') as out_f:
            pickle.dump(value, out_f)


def unpickle_variables(variable_names_to_load: list[str]) -> list[Any]:
    """
    Unpickles the variables in <variable_names_to_load> from files
    ./data/<variable>.pickle
    and returns the list of the unpickled variables.

    :param variable_names_to_load: List of variable names to unpickle
    :return: List of "values" of these variables in order and in the
        <variable_names_to_load>.
    """
    list_to_return = []
    global_path = os.path.dirname(__file__)

    for variable_name in variable_names_to_load:
        file_name = f'{variable_name}.pickle'
        path_to_file = os.path.join(global_path, "data", file_name)
        with open(path_to_file, 'rb') as in_f:
            var = pickle.load(in_f)
            list_to_return.append(var)
    return list_to_return


def _get_all_data() -> torch.Tensor:
    """
    This function generates the data for the models.
    The data is a tensor with size: (P ** 2, 4)

    Each row is an encoding of the equation of the type:
        "A + B = (A + B) % P, (mod P)"
    and it is encoded as:
        "AB=C"
        Where C == (A + B) % P,
        and "=" is encoded as a value P
    e.g. [50, 100, 113, 37] decodes as 50 + 100 = 37, mod 113

    The function uses the global P variable

    For P == 113
    >>> all_data = _get_all_data()
    >>> len(all_data)
    12769
    >>> all_data[100]
    tensor([  0, 100, 113, 100])
    """
    global P

    list_tmp = []
    for a in range(P):
        for b in range(P):
            sublist = [a, b, P, (a + b) % P]  # P is the encoding of the '='
            list_tmp.append(sublist)
    tensor_to_return = torch.tensor(list_tmp)
    return tensor_to_return


def _get_random_data(seed: int = 42,
                     generator: Optional[torch.Generator] = None) \
                    -> torch.Tensor:
    """
    A helper function that randomizes
    # >>> P = 113
    >>> r_data = _get_random_data()
    >>> len(r_data)
    12769
    """
    if generator is None:
        generator = torch.Generator().manual_seed(seed)

    all_data = _get_all_data()

    randomized_indexes = torch.randperm(len(all_data), generator=generator)
    randomized_data = all_data[randomized_indexes]

    return randomized_data


def get_train_and_test_data(train_fraction: float = 0.3) \
                                        -> tuple[torch.Tensor, torch.Tensor]:
    """
    >>> train_data, test_data = get_train_and_test_data()
    >>> len(train_data)
    3830
    >>> len(test_data)
    8939
    """
    randomized_data = _get_random_data()
    break_point = int(len(randomized_data) * train_fraction)
    train_data = randomized_data[:break_point]
    test_data = randomized_data[break_point:]
    return train_data, test_data


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


