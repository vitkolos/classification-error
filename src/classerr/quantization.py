import torch
import torch.nn.utils.prune


def lower_precision(net: torch.nn.Module, bits: int = 16, qt: str = 'asymmetric') -> torch.nn.Module:
    """lowers the precision of the network in-place"""
    if qt == 'symmetric' or qt == 'asymmetric':
        parameters = torch.hstack([p.flatten() for p in net.parameters()])
        minimum = parameters.min().item()
        maximum = parameters.max().item()
        abs_max = max(abs(minimum), abs(maximum))
        quant = QuantizerAsymmetric(minimum, maximum, bits) if qt == 'asymmetric' else QuantizerSymmetric(abs_max, bits)
        return quant.convert(net)

    if bits == 16 and qt == 'torch':
        return net.half().to(dtype=torch.get_default_dtype())

    if qt == 'prune':
        return prune(net)

    raise NotImplementedError


def prune(model: torch.nn.Module):
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            torch.nn.utils.prune.ln_structured(
                module,
                name='weight',
                amount=0.1,
                n=1,  # L1 norm is used
                dim=0,
            )
            torch.nn.utils.prune.remove(module, 'weight')

    return model


class Quantizer:
    def quant_round(self, number: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def convert(self, network: torch.nn.Module) -> torch.nn.Module:
        for p in network.parameters():
            p.copy_(self.quant_round(p))

        return network


class QuantizerAsymmetric(Quantizer):
    """uniform affine (asymmetric) quantization: maps the values of parameters onto bits and back"""

    def __init__(self, min: float, max: float, bits: int = 8):
        # two's complement range for "bits" bits (but could be also from 0 to 2**bits-1)
        self.a = - 2**(bits-1)
        self.b = 2**(bits-1) - 1

        # converts the range of bits to the original range
        self.scale = (max - min) / (self.b - self.a)
        # finds an integer where to map zero
        self.zero_point = int((max * self.a - min * self.b) / (max - min))

    def quant_round(self, number: torch.Tensor) -> torch.Tensor:
        # convert (round) to "bits" bits (such as 8 bits)
        q_number = torch.round(number / self.scale + self.zero_point)
        q_number = torch.clamp(q_number, self.a, self.b)

        # convert back to float64
        return self.scale * (q_number - self.zero_point)


class QuantizerSymmetric(Quantizer):
    """symmetric quantization: maps the values of parameters onto bits and back"""

    def __init__(self, abs_max: float, bits: int = 8):
        # two's complement range for "bits" bits
        self.a = - 2**(bits-1)
        self.b = 2**(bits-1) - 1

        # converts the range of bits to the original range
        self.scale = abs_max / self.b

    def quant_round(self, number: torch.Tensor) -> torch.Tensor:
        # convert (round) to "bits" bits (such as 8 bits)
        q_number = torch.round(number / self.scale)
        q_number = torch.clamp(q_number, self.a, self.b)

        # convert back to float64
        return self.scale * q_number
