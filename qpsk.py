import numpy as np
from scipy.stats import norm

import common as c

BITS_PER_SYMBOL = 2


class QPSK:
    """
    Class for calculating the bit error rate of QPSK modulation.
    
    This class gives theoretical as well as Monte Carlo simulated
    bit error probability of QPSK modulation for a given bit to noise
    ratio given in decibels.
    """
    bit_noise_ratio_db: float
    simulated_bit_error_rate: float

    def __init__(self, bit_noise_ratio_db):
        """Initializes an instance of this class."""
        self.bit_noise_ratio_db = float(bit_noise_ratio_db)

    def __repr__(self):
        """
        Representation of this class when using print.

        Not much practical purpose exists for this function besides
        being useful for debugging.
        """
        snr_db = f'Eb/N0 = {self.bit_noise_ratio_db}:'
        theoretical_ber = self.theoretical_bit_error_probability
        theoretical = f'Theoretical BER: 10^({np.log10(theoretical_ber):.2f})'
        if hasattr(self, 'simulated_bit_error_rate'):
            simulated_ber = self.simulated_bit_error_rate
            simulated = f'Simulated BER: 10^({np.log10(simulated_ber):.2f})'
        else:
            simulated = ''
        return ' '.join((snr_db, theoretical, simulated))

    @property
    def bit_noise_ratio(self) -> float:
        """Calculate bit noise ratio from its dB counterpart."""
        return 10**(self.bit_noise_ratio_db/10)

    @property
    def symbol_noise_ratio(self) -> float:
        """Calculate symbol noise ratio from bit noise ratio."""
        return BITS_PER_SYMBOL*self.bit_noise_ratio

    @property
    def theoretical_bit_error_probability(self) -> float:
        """
        Calculate theoretical bit error probability.

        It is possible to calculate the exact symbol error probability.
        The bit error probability is an approximation derived by
        deviding the symbol error probability by the number of bits
        per symbol.
        """
        q_function_result = 1-norm.cdf(np.sqrt(2*self.bit_noise_ratio))
        symbol_error_probability = 2*q_function_result*(1-q_function_result/2)
        return symbol_error_probability/BITS_PER_SYMBOL

    def symbol_array_to_energy_array(
            self, symbol_array: np.ndarray) -> np.ndarray:
        """
        Map symbol array to energy coordinates array.

        The mapping used is:
            - Symbol 0 (bits 00) ---> Coordinates (1, 0)
            - Symbol 1 (bits 01) ---> Coordinates (0, 1)
            - Symbol 2 (bits 10) ---> Coordinates (0, -1)
            - Symbol 3 (bits 11) ---> Coordinates (-1, 0)
        """
        energy_array = np.zeros((symbol_array.size, 2), np.int32)
        energy_array[symbol_array%3 == 0] = np.array([1, 0])
        energy_array[symbol_array%3 != 0] = np.array([0, 1])
        energy_array[symbol_array >= 2] = - energy_array[symbol_array >= 2]
        return energy_array

    def add_noise(self, energy_array: np.ndarray) -> np.ndarray:
        """Add noise to energy array."""
        noise = c.generate_noise(energy_array.size, self.symbol_noise_ratio)
        energy_array = energy_array.reshape(-1)
        return (energy_array+noise).reshape((-1, BITS_PER_SYMBOL))

    def decide_symbols(self, noisy_array: np.ndarray) -> np.ndarray:
        """Decide symbols based on noisy array."""
        zero_or_one = noisy_array[:, 0] > - noisy_array[:, 1]
        zero_or_two = noisy_array[:, 0] > noisy_array[:, 1]
        decided_symbol_array = np.zeros(noisy_array.shape[0])
        decided_symbol_array[zero_or_one & zero_or_two] = 0
        decided_symbol_array[zero_or_one & (~zero_or_two)] = 1
        decided_symbol_array[(~zero_or_one) & zero_or_two] = 2
        decided_symbol_array[(~zero_or_one) & (~zero_or_two)] = 3
        decided_symbol_array = decided_symbol_array.astype(np.int32)
        return decided_symbol_array

    def count_errors(self, symbol_array: np.ndarray,
                     decided_array: np.ndarray) -> np.ndarray:
        """Compare original with decided symbols and count errors."""
        symbol_errors = ~(decided_array == symbol_array)
        correct_symbols = symbol_array[symbol_errors]
        wrong_symbols = decided_array[symbol_errors]
        single_errors = (correct_symbols + wrong_symbols) != 3
        error_counter = np.zeros(correct_symbols.size)
        error_counter[single_errors] = 1
        error_counter[~single_errors] = 2
        return np.sum(error_counter)

    def add_phase_error(self, energy: np.ndarray, phase_error: float) -> np.ndarray:
        """Add the effects of phase error to the energy array."""
        energy_cos, energy_sin = energy[:, 0], energy[:, 1]
        error_cos, error_sin = np.cos(phase_error), np.sin(phase_error)
        energy_error = np.zeros(energy.shape)
        energy_error[:, 0] = energy_cos*error_cos - energy_sin*error_sin
        energy_error[:, 1] = energy_sin*error_cos + energy_cos*error_sin
        return energy_error

    def simulated_bit_error_probability(
            self, symbol_array: np.ndarray, /, phase_error: float = 0.0) -> float:
        """Obtain bit error probability from Monte Carlo simulation."""
        energy_array = self.symbol_array_to_energy_array(symbol_array)
        energy_array = self.add_phase_error(energy_array, phase_error)
        noisy_energy_array = self.add_noise(energy_array)
        decided_symbol_array = self.decide_symbols(noisy_energy_array)
        number_of_errors = self.count_errors(symbol_array, decided_symbol_array)
        self.simulated_bit_error_rate = number_of_errors/energy_array.size
        return self.simulated_bit_error_rate


def test():
    for snr_bit_db in range(5, 11):
        simulator = QPSK(snr_bit_db)
        symbol_array = c.random_symbol_sequence(2_000_000, BITS_PER_SYMBOL)
        simulator.simulated_bit_error_probability(symbol_array)
        difference_log = np.log10(
            (simulator.simulated_bit_error_rate/
             simulator.theoretical_bit_error_probability)
        )
        print(simulator, difference_log, sep='; Log difference: ')


if __name__ == '__main__':
    test()
