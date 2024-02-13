import pickle
import os
from typing import Any


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


# def load_model_to_pickle() -> None:
#     """Loads the model into a pickle file. To save on time presumably"""
#     # Load a model (e.g. GPT-2 Small)
#     model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")
#
#     with open('data/model_backup.pickle', 'wb') as out_f:
#         pickle.dump(model, out_f)
