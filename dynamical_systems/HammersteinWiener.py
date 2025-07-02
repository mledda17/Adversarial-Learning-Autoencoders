# It's actually the hammerstein-wiener! The class is called "LinearSystem" as the the I/O non linearities are a later addition.
import numpy as np

class HammersteinWiener:
    def __init__(self, state_size=5, non_linear_input=True, sat_value=1000, sigma=0.02):
        self.non_linear_input = non_linear_input
        self.u = -1
        self.sigma = sigma
        self.state_size = state_size
        self.sat_value = sat_value
        self.input_size = 1

        self.A = np.array([[0.7555,   -0.1991,   0.00000 ,  0.00000  ,0*-0.00909],
                           [0.25000,   0.00000,   0.00000,   0.00000,   0*0.03290],
                           [0.00000,  -0.00000,   0.00000,   0.00000,   0*0.29013],
                           [0.00000,   0.00000,   .00000,   0.00000,  0*-1.05376],
                           [0.00000,   0.00000,   0.00000,   .00000,   0*1.69967]]).T

        self.B = np.array([[-0.5,   0.,  0, 0,   0]]).T
        self.C = np.array([[  0.6993 ,  -0.4427,  0,   0,   0 ,]])


    def state_map(self, x_k, u):
        u = np.clip(u, -self.sat_value, self.sat_value)

        if u > 0 and self.non_linear_input:
            u = np.sqrt(u)

        x_k = np.dot(self.A, x_k) + np.dot(self.B, u)
        return x_k


    def output_map(self, x_k):
        y = np.dot(self.C, x_k)
        return y + 5 * np.sin(y) * int(self.non_linear_input)

    def system_dynamics(self, dim, u=-1):
        x_k = np.ones((self.state_size, 1))

        y_n = np.zeros((dim, 1))
        u_n = np.zeros((dim, self.input_size))

        noise = np.random.normal(0, 1, size=(self.input_size, dim))
        
        u = np.array([ 0.0])

        for i in range(0,dim):
            if i % 10000 == 0:
                print('.',  end='')

            u[0] = noise[0][int(i / 7)]
            u_sat = np.clip(u[0], -self.sat_value, self.sat_value)

            y_n[i] = self.output_map(x_k) * 1
            x_k = self.state_map(x_k, np.reshape(u_sat, (self.input_size, 1))) * 1

            u_n[i] = u[0]

        return y_n, u_n


    def loop(self, x_k, duk):
        y_n = np.array([])
        x_k = np.reshape(x_k,(self.state_size, 1))

        for i in range(0, len(duk)):
            u = np.array(duk[i]) + np.random.normal(0, self.sigma,(1,1)) * .0
            u = u * self.std_u + self.mean_u
            temp = self.output_map(x_k)
            temp = (temp - self.mean_y) / self.std_y
            y_n = np.append(y_n, temp) + np.random.normal(0, self.sigma, (1,1)) * .0
            x_k = self.state_map(x_k, u) * 1

        return y_n[0] * 1, x_k


    def prepare_dataset(self, training_size, validation_size):
        y_n, u_n = self.system_dynamics(training_size,True)
        y_vn, u_vn = self.system_dynamics(validation_size,True)
        self.mean_y = np.mean(y_n)
        self.mean_u = np.mean(u_n)
        self.std_y = np.std(y_n)
        self.std_u = np.std(u_n)

        y_n = (y_n-self.mean_y) / self.std_y + np.random.normal(0, self.sigma, (training_size, 1))
        y_vn = (y_vn-self.mean_y) / self.std_y + np.random.normal(0, self.sigma, (validation_size, 1))
        u_n = (u_n-self.mean_u) / self.std_u + np.random.normal(0, self.sigma, (training_size, 1))
        u_vn = (u_vn-self.mean_u) / self.std_u + np.random.normal(0, self.sigma, (validation_size, 1))

        print(y_n.shape)
        print(u_n.shape)
        print(y_vn.shape)
        print(u_vn.shape)

        return np.reshape(u_n,(training_size,1)),np.reshape(y_n,(training_size,1)),\
                    np.reshape(u_vn,(validation_size,1)),np.reshape(y_vn,(validation_size,1))