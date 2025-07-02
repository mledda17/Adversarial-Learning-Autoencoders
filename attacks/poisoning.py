import numpy as np
import matplotlib.pyplot as plt
from attacks.adversarial_attack import AdversarialAttack
import pandas as pd

class PoisoningAttack():
    def __init__(self, model, past_u, past_y, u, uk_adv, yk_real, yk_pred):
        self.model = model
        self.past_u = past_u
        self.past_y = past_y
        self.u = u
        self.uk_adv = uk_adv
        self.yk_real = yk_real
        self.yk_pred = yk_pred

    def poison_information_vector(self, i, poisoning_percentage):
        if poisoning_percentage == 5:
            if i % 10 == 0 and i < 500 :
                self.past_u = np.reshape(np.append(self.past_u, self.uk_adv)[1:], (self.model.stride_len, 1))
                self.past_y = np.reshape(np.append(self.past_y, self.yk_real)[1:], (self.model.stride_len, 1))
            elif i % 10 == 0 and i >= 500:
                self.past_u = np.reshape(np.append(self.past_u, self.u)[1:], (self.model.stride_len, 1))
                self.past_y = np.reshape(np.append(self.past_y, self.yk_pred)[1:], (self.model.stride_len, 1))
            else:
                self.past_u = np.reshape(np.append(self.past_u, self.u)[1:], (self.model.stride_len, 1))
                self.past_y = np.reshape(np.append(self.past_y, self.yk_real)[1:], (self.model.stride_len, 1))
        elif poisoning_percentage == 10:
            if i % 10 == 0:
                self.past_u = np.reshape(np.append(self.past_u, self.uk_adv)[1:], (self.model.stride_len, 1))
                self.past_y = np.reshape(np.append(self.past_y, self.yk_pred)[1:], (self.model.stride_len, 1))
            else:
                self.past_u = np.reshape(np.append(self.past_u, self.u)[1:], (self.model.stride_len, 1))
                self.past_y = np.reshape(np.append(self.past_y, self.yk_real)[1:], (self.model.stride_len, 1))
        elif poisoning_percentage == 15:
            if i % 10 == 0:
                self.past_u = np.reshape(np.append(self.past_u, self.uk_adv)[1:], (self.model.stride_len, 1))
                self.past_y = np.reshape(np.append(self.past_y, self.yk_pred)[1:], (self.model.stride_len, 1))
                self.past_y = np.reshape(np.append(self.past_y, self.yk_pred)[1:], (self.model.stride_len, 1))
            else:
                self.past_u = np.reshape(np.append(self.past_u, self.u)[1:], (self.model.stride_len, 1))
                self.past_y = np.reshape(np.append(self.past_y, self.yk_real)[1:], (self.model.stride_len, 1))
        elif poisoning_percentage == 20:
            if i % 10 == 0:
                if i == 0:
                    self.past_u[-2:] = self.uk_adv
                    self.past_y[-2:] = self.yk_pred
                else:
                    self.past_u = np.reshape(np.append(self.past_u, self.uk_adv)[1:], (self.model.stride_len, 1))
                    self.past_y = np.reshape(np.append(self.past_y, self.yk_pred)[1:], (self.model.stride_len, 1))
            elif i % 10 == 9:
                self.past_u[0] = self.uk_adv
                self.past_u[-1] = self.yk_pred
                self.past_y[0] = self.uk_adv
                self.past_y[-1] = self.yk_pred
        else:
            self.past_u = np.reshape(np.append(self.past_u, self.u)[1:], (self.model.stride_len, 1))
            self.past_y = np.reshape(np.append(self.past_y, self.yk_real)[1:], (self.model.stride_len, 1))


def poisoning_run(model, system, multi_harmonic=True, reset=-1, y_true=None,
                       u_vn=None, attack_params=None, poisoning_percentage=5):
    
    open_loop_starting_point = 15
    past_y = np.zeros((model.stride_len, 1))
    past_u = np.zeros((model.stride_len, 1))

    if y_true is None:
        x0_real_system = np.zeros((system.state_size,))

    x0 = model.encoder_network.predict([past_y.T, past_u.T])
    log_y, log_u, log_yr, log_u_adv = [], [], [], []
    final_range = 300

    if not(y_true is None):
        final_range = y_true.shape[0]

    attack = AdversarialAttack(model, epsilon=attack_params["epsilon"],
                               iterations=attack_params["iterations"])

    poisoning_attack = PoisoningAttack(model, past_u, past_y, None, None, None, None)

    for i in range(0, final_range):
        print(f"Iteration {i}")
        u = 0.5 * np.array([[np.sin(i / (20 + 0.01 * i))]]) + 0.5

        if not multi_harmonic:
            u = [u_vn[i]]

        if y_true is None:
            yk_real, x0_real_system_ = system.loop(x0_real_system, u)
            x0_real_system = np.reshape(x0_real_system_, (system.state_size,))
        else:
            yk_real = y_true[i]
            u = [u_vn[i]]

        uk_adv, yk_pred = attack.run_output(u, x0, yk_real)

        poisoning_attack.u = u
        poisoning_attack.uk_adv = uk_adv
        poisoning_attack.yk_real = yk_real
        poisoning_attack.yk_pred = yk_pred

        poisoning_attack.poison_information_vector(i, poisoning_percentage)

        past_u = poisoning_attack.past_u
        past_y = poisoning_attack.past_y


        if i < open_loop_starting_point or (i % reset == 0 and reset > 0):
            x0 = model.encoder_network.predict([past_y.T, past_u.T])
            print('*', end='')
        else:
            # Prediction of the next state based on the bridge network
            x0 = model.bridge_network.predict([uk_adv, x0])[0]

        y = model.decoder_network.predict([x0])[0]

        if i >= open_loop_starting_point:
            log_y += [y[0][-2]]
            log_yr += [yk_real[0]]
            log_u += [u[0]]
            log_u_adv += [uk_adv[0]]

        print('.', end='')

    print('\n')
    log_y = np.array(log_y).reshape(-1, 1)
    log_yr = np.array(log_yr).reshape(-1, 1)
    log_u = np.array(log_u).reshape(-1, 1)
    log_u_adv = np.array(log_u_adv).reshape(-1, 1)

    # Save single file with all relevant data for pgfplots
    df = pd.DataFrame({
        'time': np.arange(log_y.shape[0]),
        'y_true': log_yr.flatten(),
        'y_pred': log_y.flatten(),
        'u': log_u.flatten(),
        'u_adv': log_u_adv.flatten()
    })

    # Save to space-separated .dat file
    df.to_csv(f'results_for_pgfplots_poisoning_{poisoning_percentage}.dat', sep=' ', index=False)

    a = np.linalg.norm(np.array(log_y) - np.array(log_yr))
    b = np.linalg.norm(np.mean(np.array(log_yr)) - np.array(log_yr))
    fit = 1 - (a / b)
    fit = np.max([0, fit])
    print('Fit: ', fit)

    plt.figure(1, figsize=(13,10), dpi=1000)
    fit = round(fit, 4)
    plt.xlabel("Iterations", fontsize=32)
    plt.xticks(fontsize=28)
    plt.ylabel(r"$\hat{y}$, $y$", fontsize=32)
    plt.yticks(fontsize=28)
    y = plt.plot(log_y)
    yr = plt.plot(log_yr, linestyle="--")
    epsilon = str(attack.epsilon).replace(".", "")
    plt.savefig("plot_poisoning_"+str(poisoning_percentage)+".pdf", dpi=1000)
    plt.grid()
    plt.close


    return fit, log_y, log_yr