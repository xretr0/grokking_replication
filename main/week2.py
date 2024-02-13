import torch
import transformer_lens
from typing import Any, Optional


import pickle_helpers


P = 113


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
    >>> my_train_data, my_test_data = get_train_and_test_data()
    >>> len(my_train_data)
    3830
    >>> len(test_data)
    8939
    """
    randomized_data = _get_random_data()
    break_point = int(len(randomized_data) * train_fraction)
    train_data = randomized_data[:break_point]
    test_data = randomized_data[break_point:]
    return train_data, test_data


def _calc_distance_mod_p(a: int, b: int) -> int:
    ...


def _calc_loss_single_logit(prediction: int,
                            neg_log_probability: float,
                            true_value: int) -> float:
    """
    A helper function that calculates a loss of a single logit.

    :param prediction: The predicted value
    :param neg_log_probability: The negative log probability of the prediction
    :param true_value: The correct value (a+b) % P in our case
    :return: The loss of the single logit.

    >>> _calc_loss_single_logit(0, 0, 0)
    0
    >>> _calc_loss_single_logit(0, 0, 1)
    XXX
    """

    ...


def calc_loss(logits: torch.Tensor | list,
              true_value: int) -> float:
    """
    Calculates the loss of the prediction given the <logits> and
    the <true_value>.

    :param logits: A tensor of negative log probabilities of the predicitons.
        Has to be in order 0, 1, ..., P-1.

    :param true_value: The correct completion of the prompt, (a+b) % P
    :return: The total loss of this prediction.


    >>> my_tensor = torch.ones(3)
    >>> my_tensor /= len(my_tensor)

    >>> calc_loss(my_tensor, 0)
    """
    if isinstance(logits, torch.Tensor):
        logits_list = logits.tolist()
    elif isinstance(logits, list):
        logits_list = logits
    else:
        raise ValueError("The <logits> should be a Tensor or a list")

    # assert len(logits) == P

    total_loss = 0

    for prediction, neg_log_probability in enumerate(logits_list):
        additional_loss = _calc_loss_single_logit(prediction,
                                                  neg_log_probability,
                                                  true_value)
        total_loss += additional_loss

    return total_loss


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
        # d_vocab_out??
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


if __name__ == "__main__":
    main()
