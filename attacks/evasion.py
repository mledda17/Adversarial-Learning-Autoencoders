import numpy as np
import matplotlib.pyplot as plt
from attacks.adversarial_attack import AdversarialAttack
import pandas as pd

def evasion_attack(model, system, multi_harmonic=True, reset=-1, y_true=None,
                       u_vn=None, attack_params=None):
    
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

    for i in range(0, final_range):
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

        past_u = np.reshape(np.append(past_u, uk_adv)[1:],(model.stride_len, 1))
        past_y = np.reshape(np.append(past_y, yk_real)[1:],(model.stride_len, 1))

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
    diff_log_y = log_yr - log_y
    diff_log_u = log_u_adv - log_u

    # Save single file with all relevant data for pgfplots
    df = pd.DataFrame({
        'time': np.arange(log_y.shape[0]),
        'y_true': log_yr.flatten(),
        'y_pred': log_y.flatten(),
        'u': log_u.flatten(),
        'u_adv': log_u_adv.flatten(),
        'diff_y': diff_log_y.flatten(),
        'diff_u': diff_log_u.flatten()
    })

    # Save to space-separated .dat file
    df.to_csv('results_for_pgfplots.dat', sep=' ', index=False)

    a = np.linalg.norm(np.array(log_y) - np.array(log_yr))
    b = np.linalg.norm(np.mean(np.array(log_yr)) - np.array(log_yr))
    fit = 1 - (a / b)
    fit = np.max([0, fit])
    print('Fit: ', fit)

    
    # Plot real output
    plt.figure(1, figsize=(12, 10), dpi=1000)
    plt.xlabel("Iterations", fontsize=32)
    plt.ylabel(r"$y$", fontsize=32)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.plot(log_yr, color='orange')
    plt.savefig("output_real.pdf", dpi=1000)
    plt.close()

    # Plot estimated output and difference between real and estimated output
    plt.figure(2, figsize=(12,10), dpi=1000)
    plt.xlabel("Iterations", fontsize=32)
    plt.ylabel(r"$\hat{y}$", fontsize=32)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.plot(log_y)
    plt.savefig("output_estimation.pdf", dpi=1000)
    plt.close()

    # 1) Plot difference between logYR and logY
    plt.figure(3, figsize=(12,10), dpi=1000)
    plt.xlabel("Iterations", fontsize=32)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.plot(diff_log_y)
    plt.savefig("output_difference.pdf", dpi=1000)
    plt.close()

    # 2) Plot original input sequence logU
    plt.figure(4, figsize=(12,10), dpi=1000)
    plt.xlabel("Iterations", fontsize=32)
    #plt.ylabel(r"$u$", fontsize=32)
    plt.plot(log_u)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.savefig("input_original.pdf", dpi=1000)
    plt.close()

    # 3) Plot adversarial input sequence logU_adv
    plt.figure(5, figsize=(12,10), dpi=1000)
    plt.xlabel("Iterations", fontsize=32)
    #plt.ylabel(r"$\Tilde{u}$", fontsize=32)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.plot(log_u_adv)
    plt.savefig("input_adversarial.pdf", dpi=1000)
    plt.close()

    # 4) Plot difference between logU and logU_adv
    plt.figure(6, figsize=(12, 10), dpi=1000)
    plt.xlabel("Iterations", fontsize=32)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.ylim([-5, 5])
    plt.plot(diff_log_u)
    plt.savefig("input_difference.pdf", dpi=1000)
    plt.close()


    return fit, log_y, log_yr, log_u, log_u_adv, diff_log_y, diff_log_u