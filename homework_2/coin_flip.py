import numpy as np
import random

class CoinFlip:
    def __init__(self, coins, flips, repeats):
        self.coins = coins
        self.flips = flips
        self.repeats = repeats
        self.finalResult = {
            "nu_1" : [],
            "nu_min" : [],
            "nu_rand" : []
            }

    def coin_flip(self):
        result = []
        for i in range(self.flips):
            heads = 0
            for j in range(self.coins):
                bit = int(random.getrandbits(1))
                if bit == 0: heads += 1
            result.append(heads)
        return result

    def experiment_run(self):
        for i in range(self.repeats):
            result_init = self.coin_flip() 
            self.finalResult.get("nu_1").append(result_init[0]/10.)
            self.finalResult.get("nu_min").append(min(result_init)/10.)
            self.finalResult.get("nu_rand").append(result_init[random.randint(0, self.flips-1)]/10.)
        return np.sum(self.finalResult.get("nu_min")) / float(len(self.finalResult.get("nu_min")))

if __name__ == "__main__":
    # problem 1
    run1 = CoinFlip(10,1000,100000)
    print(run1.experiment_run())
    # returns the value of 0.0378 answer [b]

    #problem 2
    print(run1.finalResult())





