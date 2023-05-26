import torch


class BiaffineAttention(torch.nn.Module):
    # `fixed-class biaffine attention`
    """Implements a biaffine attention operator for binary relation classification.
    PyTorch implementation of the biaffine attention operator from "End-to-end neural relation
    extraction using deep biaffine attention" (https://arxiv.org/abs/1812.11275) which can be used
    as a classifier for binary relation classification.
    Args:
        in_features (int): The size of the feature dimension of the inputs.
        out_features (int): The size of the feature dimension of the output.
    Shape:
        - x_1: `(N, *, in_features)` where `N` is the batch dimension and `*` means any number of
          additional dimensisons.
        - x_2: `(N, *, in_features)`, where `N` is the batch dimension and `*` means any number of
          additional dimensions.
        - Output: `(N, *, out_features)`, where `N` is the batch dimension and `*` means any number
            of additional dimensions.
    Examples:
        >>> batch_size, in_features, out_features = 32, 100, 4
        >>> biaffine_attention = BiaffineAttention(in_features, out_features)
        >>> x_1 = torch.randn(batch_size, in_features)
        >>> x_2 = torch.randn(batch_size, in_features)
        >>> output = biaffine_attention(x_1, x_2)
        >>> print(output.size())
        torch.Size([32, 4])
    """
    def __init__(self, head_in_features, tail_in_features, out_features):
        super(BiaffineAttention, self).__init__()

        # self.in_features = in_features
        self.head_in_features = head_in_features
        self.tail_in_features = tail_in_features
        self.out_features = out_features

        self.bilinear = torch.nn.Bilinear(head_in_features, tail_in_features, out_features, bias=False)
        self.linear = torch.nn.Linear(head_in_features + tail_in_features, out_features, bias=True)

        self.reset_parameters()  # use default parameter initialization tricks implemented by PyTorch

        print("initialized BiaffineAttention")

    def forward(self, x_1, x_2):
        return self.bilinear(x_1, x_2) + self.linear(torch.cat((x_1, x_2), dim=-1))

    def reset_parameters(self):
        self.bilinear.reset_parameters()
        self.linear.reset_parameters()


class BiaffinePointerAttention(torch.nn.Module):
    # `variable-class biaffine attention`
    # 12/13
    """Implements a biaffine attention operator for binary relation classification.
    PyTorch implementation of the biaffine attention operator from "End-to-end neural relation
    extraction using deep biaffine attention" (https://arxiv.org/abs/1812.11275) which can be used
    as a classifier for binary relation classification.
    Args:
        in_features (int): The size of the feature dimension of the inputs.
        out_features (int): The size of the feature dimension of the output.
    Shape:
        - x_1: `(N, *, in_features)` where `N` is the batch dimension and `*` means any number of
          additional dimensisons.
        - x_2: `(N, *, in_features)`, where `N` is the batch dimension and `*` means any number of
          additional dimensions.
        - Output: `(N, *, out_features)`, where `N` is the batch dimension and `*` means any number
            of additional dimensions.
    Examples:
        >>> batch_size, in_features, out_features = 32, 100, 4
        >>> biaffine_attention = BiaffineAttention(in_features, out_features)
        >>> x_1 = torch.randn(batch_size, in_features)
        >>> x_2 = torch.randn(batch_size, in_features)
        >>> output = biaffine_attention(x_1, x_2)
        >>> print(output.size())
        torch.Size([32, 4])
    """
    def __init__(self, head_in_features, tail_in_features, out_features):
        super(BiaffinePointerAttention, self).__init__()

        # self.in_features = in_features
        self.head_in_features = head_in_features
        self.tail_in_features = tail_in_features
        self.out_features = out_features

        self.bilinear = torch.nn.Bilinear(head_in_features, tail_in_features, out_features, bias=False)
        self.linear = torch.nn.Linear(head_in_features, out_features, bias=True)

        self.reset_parameters()  # use default parameter initialization tricks implemented by PyTorch

        print("initialized BiaffinePointerAttention")

    def forward(self, x_1, x_2):
        return self.bilinear(x_1, x_2) + self.linear(x_1)

    def reset_parameters(self):
        self.bilinear.reset_parameters()
        self.linear.reset_parameters()