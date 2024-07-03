import numpy as np
from scipy.stats import ncx2
from scipy.special import i0

import common as c
from qpsk import QPSK

BITS_PER_SYMBOL = 2


class DQPSK(QPSK):
    """
    Class for calculating the bit error rate of DQPSK modulation.
    
    This class gives theoretical as well as Monte Carlo simulated
    bit error probability of DQPSK modulation for a given bit to noise
    ratio given in decibels.
    """
    @property
    def theoretical_bit_error_probability(self) -> float:
        """Calculate theoretical bit error probability."""
        q_function_a = 2*self.bit_noise_ratio*(1-np.sqrt(1/2))
        q_function_b = 2*self.bit_noise_ratio*(1+np.sqrt(1/2))
        q_function_result = 1-ncx2.cdf(df=2, nc=q_function_a, x=q_function_b)
        bessel_result = i0(np.sqrt(q_function_a*q_function_b))
        bessel_factor = (1/2)*np.exp(-(q_function_a+q_function_b)/2)
        return q_function_result-bessel_result*bessel_factor

    def symbol_to_angle(self, symbol_array: np.ndarray) -> np.ndarray:
        """
        Obtain angle array from symbol array.
        
        Angles are represented by the factor multiplying pi/2, i.e.,
        an angle of k*pi/2 is represented by k (mod 4).
        """
        angle_array = np.zeros(symbol_array.size)
        angle_array[symbol_array == 1] = 1
        angle_array[symbol_array == 2] = 3
        angle_array[symbol_array == 3] = 2
        return np.cumsum(angle_array) % 4

    def get_symbol_angles(self, angle_array: np.ndarray) -> np.ndarray:
        """Convert transmited angles to symbol angles."""
        symbol_angles = np.zeros(angle_array.size)
        symbol_angles[0] = angle_array[0]
        symbol_angles[1:] = (angle_array[1:] - angle_array[:-1])
        return symbol_angles

    def symbol_angle_to_symbol(self, symbol_angles: np.ndarray) -> np.ndarray:
        """Obtain symbols from their corresponding angles."""
        symbol_array = np.zeros(symbol_angles.size)
        symbol_array[symbol_angles == 1] = 1
        symbol_array[symbol_angles == 2] = 3
        symbol_array[symbol_angles == 3] = 2
        return symbol_array.astype(np.int32)

    def transmission_angle_to_symbol(self, angle_array: np.ndarray) -> np.ndarray:
        """Convert angle array to symbol array."""
        symbol_angles = self.get_symbol_angles(angle_array) % 4
        return self.symbol_angle_to_symbol(symbol_angles)

    def angle_to_energy(self, angle: np.ndarray) -> np.ndarray:
        """
        Obtain energy array from angle array.
        
        Important to note that this method repurposes the method
        symbol_array_to_energy_array.
        """
        angle[angle == 2] = -1
        angle[angle == 3] = 2
        angle[angle == -1] = 3
        return self.symbol_array_to_energy_array(angle)

    def decide_symbols_dpsk_coherent(self, noisy_energy: np.ndarray) -> np.ndarray:
        """
        Decide symbols from noisy array for coherent DQPSK.
        
        This method repurposes the method decide_symbols to decide
        angles instead.
        """
        decided_angle_array = self.decide_symbols(noisy_energy)
        decided_angle_array[decided_angle_array == 2] = -1
        decided_angle_array[decided_angle_array == 3] = 2
        decided_angle_array[decided_angle_array == -1] = 3
        decided_symbol_array = self.transmission_angle_to_symbol(decided_angle_array)
        return decided_symbol_array

    def decide_symbol_angles_non_coherent(self, noisy_energy: np.ndarray) -> np.ndarray:
        """Decide angles from noisy array for non-coherent DQPSK."""
        transmission_angle = np.arctan2(noisy_energy[:, 1], noisy_energy[:, 0])
        symbol_angles = self.get_symbol_angles(transmission_angle)
        return np.round(symbol_angles/(np.pi/2)).astype(np.int32) % 4

    def decide_symbols_dpsk_non_coherent(self, noisy_energy: np.ndarray) -> np.ndarray:
        """Decide symbols from noisy array for non-coherent DQPSK."""
        decided_angle_array = self.decide_symbol_angles_non_coherent(noisy_energy)
        decided_symbol_array = self.symbol_angle_to_symbol(decided_angle_array)
        return decided_symbol_array

    def simulated_bit_error_probability(
            self, symbol_array: np.ndarray, /,
            coherent: bool, phase_error: float = 0.0) -> float:
        """Obtain bit error probability from Monte Carlo simulation."""
        angle_array = self.symbol_to_angle(symbol_array)
        energy_array = self.angle_to_energy(angle_array)
        energy_array = self.add_phase_error(energy_array, phase_error)
        noisy_energy_array = self.add_noise(energy_array)
        if coherent:
            decided_symbol_array = self.decide_symbols_dpsk_coherent(noisy_energy_array)
        else:
            decided_symbol_array = self.decide_symbols_dpsk_non_coherent(noisy_energy_array)
        number_of_errors = self.count_errors(symbol_array, decided_symbol_array)
        self.simulated_bit_error_rate = number_of_errors/energy_array.size
        return self.simulated_bit_error_rate


def main():
    for snr_bit_db in range(5, 11):
        simulator = DQPSK(snr_bit_db)
        symbol_array = c.random_symbol_sequence(2_000_000, BITS_PER_SYMBOL)
        simulator.simulated_bit_error_probability(symbol_array, coherent=True, phase_error=np.pi/2)
        difference_log = np.log10(
            (simulator.simulated_bit_error_rate/
             simulator.theoretical_bit_error_probability)
        )
        print(simulator, difference_log, sep='; Log difference: ')


if __name__ == '__main__':
    main()
