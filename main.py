from utilities.systemselector import SystemSelectorEnum
from utilities.train_autoencoder import train_autoencoder
from validation.open_loop_val import open_loop_validation
from attacks.evasion import evasion_attack
from attacks.poisoning import poisoning_run
            

if __name__ == '__main__':
    sys = SystemSelectorEnum.twotanks

    weights_path = f"src/weights/twotanks.weights.h5"
    
    model, system, u_n, y_n, u_vn, y_vn = train_autoencoder(
        fit_horizon=5, state_size=6, pairs_iv=10, regularizer_weight=1e-4, output_window_len=2,
        n_layer=3, n_neurons=30, train_ann=True, weights_path=weights_path, sys=sys
    )

    print("Attack Scenario 1")

    epsilons = [0.01, 0.1, 0.3, 0.5]
    n_iterations = [1, 10, 30]

    validation_params = dict(VoM=False, reset=10)
    
    for e in epsilons:
        for n in n_iterations:
            attack_params = dict(epsilon=e, iterations=n)

            fit, log_y, log_yr, log_u, log_u_adv, diff_log_y, diff_log_u = evasion_attack(
                model, system, multi_harmonic=validation_params["VoM"], reset=validation_params["reset"],
                y_true=None, u_vn=u_vn.copy(), attack_params=attack_params
            )

    print("Attack Scenario 2")

    epsilons = [0.01]
    n_iterations = [1]
    poison_p = [5, 10, 15, 20]

    # VoM is a parameter to set the nature of the input signal.
    # True means setting an input signal with multi-harmonic nature.
    # False means setting a casual input signal.
    validation_params = dict(VoM=True, reset=10)

    for p in poison_p:
        for e in epsilons:
            for n in n_iterations:
                attack_params = dict(epsilon=e, iterations=n)

                fit, log_y, log_yr = poisoning_run(model, system,
                                                        multi_harmonic=validation_params["VoM"], reset=validation_params["reset"],
                                                        y_true=None, u_vn=u_vn.copy(), attack_params=attack_params, poisoning_percentage=p)

    
                    
                

        