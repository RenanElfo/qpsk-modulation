import numpy as np

SYMBOL_ENERGY = 1


def random_symbol_sequence(
        size: int, bits_per_symbol: int,
        probability_of_zero: float = 0.5) -> np.ndarray:
    """Generate random sequence of symbols of given size."""
    bits = random_bit_sequence(size, probability_of_zero)
    return bit_array_to_symbol_array(bits, bits_per_symbol)


def random_bit_sequence(
        size: int, probability_of_zero: float = 0.5) -> np.ndarray:
    """Generate random sequence of bits of given size."""
    return (np.random.rand(size) > probability_of_zero).astype(np.int32)


def generate_noise(size: int, symbol_noise_ratio: float) -> np.ndarray:
    """Generate sequence of gaussian noise of given size."""
    noise_spectral_density = SYMBOL_ENERGY/symbol_noise_ratio
    noise_variance = noise_spectral_density/2
    standard_deviation = np.sqrt(noise_variance)
    return np.random.normal(scale=standard_deviation, size=size)


def bit_array_reshapen(
        bit_array: np.ndarray, bits_per_symbol: int) -> np.ndarray:
    """
    Reshape sequence of bits to represent symbols.

    Reshape sequence of bits such that each row contains the bits
    corresponding to a symbol.
    """
    if bit_array.size%bits_per_symbol != 0:
        message = 'Array size not a multiple of bits per symbol value.'
        raise ValueError(message)
    reshapen_array = np.reshape(
        bit_array, (bit_array.size//bits_per_symbol, bits_per_symbol)
    )
    return reshapen_array


def bit_array_to_symbol_array(
        bit_array: np.ndarray, bits_per_symbol: int) -> np.ndarray:
    """
    Convert sequence of bits into sequence of symbols.

    Each group of bits_per_symbol bits is converted into an integer
    from 0 to 2^(bits_per_symbol)-1. If bit_array has size not multiple
    of bits_per_symbol, an error is raised.
    """
    reshapen_array = bit_array_reshapen(bit_array, bits_per_symbol)
    return bit_array_to_int(reshapen_array)


def bit_array_to_int(bit_array: np.ndarray) -> int:
    """
    Convert array of bits into an integer.

    The bit array is interpreted as an integer in base 2 and is
    therefore returned as an integer in base 10.
    """
    return np.dot(bit_array, np.flip(1 << np.arange(bit_array.shape[1])))


def test():
    pass
    # print(random_sequence(1000, probability_of_zero=0.99))
    # bit_array = np.array([1, 2, 3, 4, 5, 6])
    # bit_array_to_symbol_array(bit_array, 2)
    #print(bit_array_to_int(np.array([1, 1, 0, 0])))
    #print(bit_array_to_symbol_array(np.array([0, 0, 0, 1, 1, 1, 1, 0]), 2))
    #symbol_noise_ratio_db = 6
    #symbol_noise_ratio = 10**(symbol_noise_ratio_db/10)
    #print(np.max(generate_noise(20_000_000, symbol_noise_ratio)))
    z = random_bit_sequence(20_000_000)
    print(z)
    x = bit_array_reshapen(z, 2)
    print(x)
    y = np.flip(1 << np.arange(x.shape[1]))
    print(a := np.dot(x, y), b := bit_array_to_symbol_array(z, 2), np.all(a == b))



if __name__ == '__main__':
    test()
