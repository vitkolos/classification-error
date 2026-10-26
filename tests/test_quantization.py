import torch
from classerr.quantization import QuantizerAsymmetric

tt = torch.tensor

def test_quantization():
    q = QuantizerAsymmetric(-1.5, 1.5, bits=2)
    assert torch.equal(q.quant_round(tt(0)), tt(0)), 'zero should be zero even after quantization'
    assert torch.equal(q.quant_round(tt(0.803)), q.quant_round(tt(0.804))), 'number close to each other should be quantized to the same number'
    assert torch.equal(q.quant_round(tt(5)), q.quant_round(tt(6))), 'numbers out of bounds should be quantized to the same number'
    range_ = torch.arange(-1.5, 1.6, 0.1)
    assert len(set(q.quant_round(range_).tolist())) == 4, 'there should be only four (2**bits) unique values after quantization'
