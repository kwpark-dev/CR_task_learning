import torch
import torch.nn as nn
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# from glob import glob

from models.network import StackingBlock
from models.random_process import GPRegression
from models.order_table_fixed import table


if __name__ == '__main__':
    freq_table = {'target_1':[0, 0, 0], 'target_2':[0, 0, 0], 'target_3':[0,0,0]}
    objects = ['target_1', 'target_2', 'target_3']
    trial = 10

    for i in range(10):
        order = list(table[i].keys())
        
        for j in objects:
            idx = order.index(j)
            freq_table[j][idx] += 1

    freq_table = np.array(list(freq_table.values()))
    idx = np.argmax(freq_table, axis=1) # solid way
    
    order_graph = np.array(objects)[idx] # machine should notice which block first, second, etc
    print(order_graph)
    start = []
    pass_through = []

    for i in range(10):
        primary = order_graph[0] 
        start.append(table[i][primary])
        
        sec = order_graph[1]
        thrd = order_graph[-1]
        pass_tmp = table[i][sec] + table[i][thrd]
        pass_through.append(pass_tmp)


    start = torch.from_numpy(np.array(start)).float()
    pass_through = torch.from_numpy(np.array(pass_through)).float()
    
    input_node = torch.tensor([3])
    hidden_node = torch.tensor([6, 12, 6])
    output_node = torch.tensor([6])
    stack_net = StackingBlock(input_node, hidden_node, output_node)
    
    epochs = 40000

    loss_func = nn.MSELoss()
    optimizer = optim.Adam(stack_net.parameters())

    loss_train = []

    for i in tqdm(range(epochs)):
        optimizer.zero_grad()
            
        position = stack_net(start)
        loss = loss_func(position, pass_through)
        loss_train.append(loss.item())

        loss.backward()
        optimizer.step()

    stack_net.eval()

    primary_end = torch.tensor([-0.1352, 0.32341, 0.225]).float()
    poses = stack_net(primary_end) # highly likely overfit. anyhow it returns position for target 2 and 3 so that robot stack them properly.
    poses = poses.reshape(2, -1).detach().numpy()
    print(poses.reshape(2, -1))

    # plt.plot(loss_train)
    # plt.show()
    primary_end = primary_end.detach().numpy()

    initial_xy = np.random.uniform(-0.3, 0.5, (2, 2)) # assume tha detected position
    # initial_z = np.random.uniform(0.225, 0.25, (2,1))
    initial_z = np.array([[0.225], [0.225]])
    initial_dist = np.hstack((initial_xy, initial_z))
    # print(initial_dist)

    train_coords = np.array([primary_end,
                             initial_dist[0], poses[0],
                             initial_dist[1], poses[1]])

    time = np.arange(0, 50, 10)
    
    n_seq = 300
    n_traj = 3
    gp = GPRegression(time, train_coords, 0, 40)
    # gpx, gpy, gpz = gp.get_traj(n_seq, n_traj=5)
    est_traj = gp.get_traj(n_seq, n_traj)
    
    
    # col_traj = np.genfromtxt('joint_traj_sample/SAMPLE_3/common_end_effector_link.csv', delimiter=',', skip_header=1)
    # # print(col_traj)
    # _, x, y, z, _, _, _ = col_traj.T

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for i in range(n_traj):
        gpx, gpy, gpz = est_traj[i]
        ax.plot3D(gpx, gpy, gpz, label='GP trajectory_'+str(i))

    ax.scatter(train_coords.T[0][[1,3]], train_coords.T[1][[1,3]], train_coords.T[2][[1,3]], color='red', label='initial positions')
    ax.scatter(train_coords.T[0][[0,2,-1]], train_coords.T[1][[0,2,-1]], train_coords.T[2][[0,2,-1]], color='green', label='stack positions')
    # ax.plot3D(x, y, z, color='blue', label='sampled trajectory')
    ax.legend()
    plt.suptitle('Pass through')

    plt.show()