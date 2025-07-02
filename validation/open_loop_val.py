import numpy as np
import matplotlib.pyplot as plt

def open_loop_validation(model, system, multi_harmonic=True, y_true=None, u_vn=None):    
    starting_point = 15
    past_y = np.zeros((model.stride_len, 1))
    past_u = np.zeros((model.stride_len,1))
    
    if y_true is None:
        x0_real_system = np.zeros((system.state_size, ))
        
    x0 = model.encoder_network.predict([past_y.T, past_u.T])

    log_y = []
    log_u = []
    log_y_real = []

    final_range = 500

    if not(y_true is None):
        final_range = y_true.shape[0]

    for i in range(0, final_range):
        print(f"Step: {i}/{final_range}")
        u = 0.5 * np.array([[np.sin(i / (20 + 0.01 * i))]]) + 0.5

        if not multi_harmonic:
            u = [u_vn[i]]

        if y_true is None:
            y_k_real, x0_real_system_ = system.loop(x0_real_system, u)
            x0_real_system = np.reshape(x0_real_system_, (system.state_size,))
        else:
            y_k_real = y_true[i]
            u = [u_vn[i]]
        
        past_u = np.reshape(np.append(past_u, u)[1:],(model.stride_len, 1))
        past_y = np.reshape(np.append(past_y, y_k_real)[1:],(model.stride_len, 1))

        x0 = model.encoder_network.predict([past_y.T, past_u.T])

        y = model.decoder_network.predict([x0])[0]

        if i >= starting_point:
            log_y += [y[0][-2]]
            log_y_real += [y_k_real[0]]
            log_u += [u[0]]
        
    print('\n')
    log_y = np.array(log_y).reshape((-1, 1))
    log_y_real = np.array(log_y_real).reshape((-1, 1))  

    a = np.linalg.norm(np.array(log_y)-np.array(log_y_real))
    b = np.linalg.norm(np.mean(np.array(log_y_real))-np.array(log_y_real))

    fit = 1 - (a / b)
    fit = np.max([0, fit])

    print('Fit: ', fit)

    plt.figure()
    plt.title("Best Fit Ratio = " + str(fit))
    y = plt.plot(log_y, label=r"$\hat{y}$")
    yr = plt.plot(log_y_real, label=r"$y$")
    plt.legend()
    plt.show()

    return fit, log_y, log_y_real